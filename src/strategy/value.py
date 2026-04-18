"""
Value detection — identifies bets where blended model-market probability
exceeds the market line, with confirmation from both models.

Strategy: Anchor on market odds (they're well-calibrated) and only adjust
where the model has high conviction AND the no-odds model independently agrees.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    MIN_EDGE_THRESHOLD,
    NEAR_MISS_MIN_EDGE,
    MIN_MODEL_PROB,
    MIN_FIGHTER_FIGHTS,
    NEWBIE_FEEDER_EXTRA_EDGE,
    NEWBIE_FIGHTS_THRESHOLD,
    NEWBIE_MAJOR_EXTRA_EDGE,
    NEWBIE_REGIONAL_EXTRA_EDGE,
    NEWBIE_SIZE_MULTIPLIER,
    NEWBIE_SKIP_ZERO_FIGHTS_REGIONAL,
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


@dataclass(frozen=True)
class NewbieRuleConfig:
    """Configuration for handling low-experience UFC fighters."""

    name: str
    mode: str = "hard_skip"
    min_fighter_fights: int = MIN_FIGHTER_FIGHTS
    threshold: int = NEWBIE_FIGHTS_THRESHOLD
    major_extra_edge: float = NEWBIE_MAJOR_EXTRA_EDGE
    feeder_extra_edge: float = NEWBIE_FEEDER_EXTRA_EDGE
    regional_extra_edge: float = NEWBIE_REGIONAL_EXTRA_EDGE
    size_multiplier: float = NEWBIE_SIZE_MULTIPLIER
    skip_zero_fights_regional: bool = NEWBIE_SKIP_ZERO_FIGHTS_REGIONAL


@dataclass(frozen=True)
class NewbieAdjustment:
    """Resolved fight-level experience penalty applied by the decision layer."""

    allowed: bool
    extra_edge_required: float = 0.0
    size_multiplier: float = 1.0
    is_newbie_bet: bool = False
    reason: str = ""


DEFAULT_NEWBIE_RULE = NewbieRuleConfig(
    name=f"hard_skip_min_{MIN_FIGHTER_FIGHTS}",
    mode="hard_skip",
    min_fighter_fights=MIN_FIGHTER_FIGHTS,
    threshold=NEWBIE_FIGHTS_THRESHOLD,
)

LEGACY_BASELINE_NEWBIE_RULE = NewbieRuleConfig(
    name="hard_skip_min_3",
    mode="hard_skip",
    min_fighter_fights=3,
    threshold=3,
)

THRESHOLD_DROP_NEWBIE_RULE = NewbieRuleConfig(
    name="hard_skip_min_2",
    mode="hard_skip",
    min_fighter_fights=2,
    threshold=3,
)

TIERED_ORG_AWARE_NEWBIE_RULE = NewbieRuleConfig(
    name="tiered_org_aware",
    mode="tiered",
    min_fighter_fights=MIN_FIGHTER_FIGHTS,
    threshold=NEWBIE_FIGHTS_THRESHOLD,
)


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


def _coerce_probability(value: object) -> Optional[float]:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(prob):
        return None
    return prob


def _first_valid_probability(*values: object, default: float = 0.5) -> float:
    for value in values:
        prob = _coerce_probability(value)
        if prob is not None:
            return prob
    logger.warning("All probability values invalid %r — falling back to %.2f", values, default)
    return default


def _coerce_optional_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _coerce_optional_int(value: object) -> Optional[int]:
    numeric = _coerce_optional_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _normalize_org_tier(org_tier: object) -> Optional[int]:
    numeric = _coerce_optional_float(org_tier)
    if numeric is None:
        return None
    if numeric <= 1.5:
        return 1
    if numeric <= 2.5:
        return 2
    return 3


def _newbie_rule_config(newbie_rule: Optional[NewbieRuleConfig]) -> NewbieRuleConfig:
    return newbie_rule or DEFAULT_NEWBIE_RULE


def _fighter_newbie_adjustment(
    *,
    num_fights: Optional[int],
    org_tier: object,
    fighter_label: str,
    newbie_rule: NewbieRuleConfig,
) -> NewbieAdjustment:
    fights = _coerce_optional_int(num_fights)
    fights = fights if fights is not None else 0

    if newbie_rule.mode != "tiered":
        if fights < newbie_rule.min_fighter_fights:
            return NewbieAdjustment(
                allowed=False,
                reason=(
                    f"{fighter_label} has only {fights} UFC fights "
                    f"(minimum: {newbie_rule.min_fighter_fights})"
                ),
            )
        is_newbie_bet = fights < newbie_rule.threshold
        reason = (
            f"{fighter_label} has {fights} UFC fights"
            if is_newbie_bet
            else ""
        )
        return NewbieAdjustment(
            allowed=True,
            is_newbie_bet=is_newbie_bet,
            reason=reason,
        )

    if fights >= newbie_rule.threshold:
        return NewbieAdjustment(allowed=True)

    tier = _normalize_org_tier(org_tier)
    tier_label = {
        1: "major",
        2: "feeder",
        3: "regional",
        None: "unknown",
    }[tier]

    if fights <= 0:
        if tier == 1:
            return NewbieAdjustment(
                allowed=True,
                extra_edge_required=newbie_rule.major_extra_edge,
                size_multiplier=newbie_rule.size_multiplier,
                is_newbie_bet=True,
                reason=f"{fighter_label} debut with major-org background",
            )
        if tier == 2:
            return NewbieAdjustment(
                allowed=True,
                extra_edge_required=newbie_rule.feeder_extra_edge,
                size_multiplier=newbie_rule.size_multiplier,
                is_newbie_bet=True,
                reason=f"{fighter_label} debut with feeder-org background",
            )
        if newbie_rule.skip_zero_fights_regional:
            return NewbieAdjustment(
                allowed=False,
                reason=f"{fighter_label} debut with {tier_label} pre-UFC background",
            )
        return NewbieAdjustment(
            allowed=True,
            extra_edge_required=newbie_rule.regional_extra_edge,
            size_multiplier=newbie_rule.size_multiplier,
            is_newbie_bet=True,
            reason=f"{fighter_label} debut with {tier_label} pre-UFC background",
        )

    if tier == 1:
        extra_edge = newbie_rule.major_extra_edge
    elif tier == 2:
        extra_edge = newbie_rule.feeder_extra_edge
    else:
        extra_edge = newbie_rule.regional_extra_edge

    return NewbieAdjustment(
        allowed=True,
        extra_edge_required=extra_edge,
        size_multiplier=newbie_rule.size_multiplier,
        is_newbie_bet=True,
        reason=f"{fighter_label} has {fights} UFC fights ({tier_label} pre-UFC background)",
    )


def newbie_penalty(
    num_fights_a: Optional[int],
    num_fights_b: Optional[int],
    org_tier_a: object = None,
    org_tier_b: object = None,
    newbie_rule: Optional[NewbieRuleConfig] = None,
) -> NewbieAdjustment:
    """
    Resolve the fight-level newbie penalty.

    The stricter of the two fighters governs the whole fight.
    """
    rule = _newbie_rule_config(newbie_rule)
    fighter_adjustments = [
        _fighter_newbie_adjustment(
            num_fights=num_fights_a,
            org_tier=org_tier_a,
            fighter_label="fighter A",
            newbie_rule=rule,
        ),
        _fighter_newbie_adjustment(
            num_fights=num_fights_b,
            org_tier=org_tier_b,
            fighter_label="fighter B",
            newbie_rule=rule,
        ),
    ]

    for adjustment in fighter_adjustments:
        if not adjustment.allowed:
            return adjustment

    extra_edge_required = max(adj.extra_edge_required for adj in fighter_adjustments)
    size_multiplier = min(adj.size_multiplier for adj in fighter_adjustments)
    newbie_reasons = [adj.reason for adj in fighter_adjustments if adj.is_newbie_bet and adj.reason]

    return NewbieAdjustment(
        allowed=True,
        extra_edge_required=extra_edge_required,
        size_multiplier=size_multiplier,
        is_newbie_bet=bool(newbie_reasons),
        reason="; ".join(newbie_reasons),
    )


def required_edge_for_market(
    market_prob: float,
    *,
    edge_scaling_base: Optional[float] = None,
    newbie_adjustment: Optional[NewbieAdjustment] = None,
) -> float:
    """Return the total edge required after odds scaling and newbie penalties."""
    decimal_odds = implied_prob_to_decimal_odds(market_prob)
    required_edge = scaled_min_edge(decimal_odds, base=edge_scaling_base)
    if newbie_adjustment is not None:
        required_edge += newbie_adjustment.extra_edge_required
    return required_edge


def compute_independent_blend_probs(
    model_a: float,
    market_a: float,
    no_odds_a: Optional[float],
    model_b: float,
    market_b: float,
    no_odds_b: Optional[float],
    base_weight: float = BLEND_WEIGHT,
) -> tuple[float, float]:
    """Blend both sides independently, then renormalize to a proper market pair."""
    dyn_weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a, base_weight)
    dyn_weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b, base_weight)
    raw_blend_a = blend_probability(model_a, market_a, dyn_weight_a)
    raw_blend_b = blend_probability(model_b, market_b, dyn_weight_b)
    total = raw_blend_a + raw_blend_b
    if total > 0:
        return raw_blend_a / total, raw_blend_b / total
    return 0.5, 0.5


def scaled_min_edge(decimal_odds: float, base: Optional[float] = None) -> float:
    """
    Calculate the minimum edge required based on odds magnitude.

    At even money (2.0), requires the base edge (3%).
    For each 1.0 increase in odds, requires an additional 2% edge.
    """
    base = base if base is not None else EDGE_SCALING_BASE
    if decimal_odds <= 2.0:
        return base
    return base + EDGE_SCALING_RATE * (decimal_odds - 2.0)


def _passes_quality_filters(
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
    a_org_tier: object = None,
    b_org_tier: object = None,
    newbie_rule: Optional[NewbieRuleConfig] = None,
    newbie_adjustment: Optional[NewbieAdjustment] = None,
    edge_scaling_base: Optional[float] = None,
    require_model_agreement: Optional[bool] = None,
    model_agreement_min_edge: Optional[float] = None,
    max_decimal_odds: Optional[float] = None,
) -> bool:
    """Check all quality filters EXCEPT the edge threshold."""
    decimal_odds = implied_prob_to_decimal_odds(market_prob)
    max_decimal_odds = MAX_DECIMAL_ODDS if max_decimal_odds is None else max_decimal_odds
    require_model_agreement = (
        REQUIRE_MODEL_AGREEMENT
        if require_model_agreement is None
        else require_model_agreement
    )
    model_agreement_min_edge = (
        MODEL_AGREEMENT_MIN_EDGE
        if model_agreement_min_edge is None
        else model_agreement_min_edge
    )
    newbie_adjustment = newbie_adjustment or newbie_penalty(
        a_num_fights,
        b_num_fights,
        org_tier_a=a_org_tier,
        org_tier_b=b_org_tier,
        newbie_rule=newbie_rule,
    )
    if not newbie_adjustment.allowed:
        logger.debug("Skipping %s: %s", fighter_name, newbie_adjustment.reason)
        return False

    # Filter 1: Minimum blended probability
    if blended_prob < MIN_MODEL_PROB:
        logger.debug(
            f"Skipping {fighter_name}: blended prob {blended_prob:.1%} below "
            f"minimum {MIN_MODEL_PROB:.1%}"
        )
        return False

    # Filter 2: Maximum odds cap
    if decimal_odds > max_decimal_odds:
        logger.debug(
            f"Skipping {fighter_name}: odds {decimal_odds:.2f} exceed "
            f"maximum {max_decimal_odds:.1f}"
        )
        return False

    # Filter 4: Model agreement — no-odds model must independently agree
    if require_model_agreement:
        if no_odds_prob is None:
            logger.debug(
                f"Skipping {fighter_name}: no-odds model probability unavailable "
                f"(require_model_agreement is enabled)"
            )
            return False
        no_odds_edge = no_odds_prob - market_prob
        if no_odds_edge < model_agreement_min_edge:
            logger.debug(
                f"Skipping {fighter_name}: no-odds model disagrees "
                f"(no-odds edge {no_odds_edge:.1%} < {model_agreement_min_edge:.1%})"
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
            required_edge = required_edge_for_market(
                market_prob,
                edge_scaling_base=edge_scaling_base,
                newbie_adjustment=newbie_adjustment,
            )
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
    a_org_tier: object = None,
    b_org_tier: object = None,
    newbie_rule: Optional[NewbieRuleConfig] = None,
    newbie_adjustment: Optional[NewbieAdjustment] = None,
    edge_scaling_base: Optional[float] = None,
    require_model_agreement: Optional[bool] = None,
    model_agreement_min_edge: Optional[float] = None,
    max_decimal_odds: Optional[float] = None,
) -> bool:
    """Check if a potential bet passes all filters (quality + edge threshold)."""
    newbie_adjustment = newbie_adjustment or newbie_penalty(
        a_num_fights,
        b_num_fights,
        org_tier_a=a_org_tier,
        org_tier_b=b_org_tier,
        newbie_rule=newbie_rule,
    )
    if not _passes_quality_filters(
        blended_prob, market_prob, edge, fighter_name, no_odds_prob,
        line_movement, line_is_sharp, line_steam_move, bet_side,
        a_num_fights, b_num_fights, a_org_tier, b_org_tier,
        newbie_rule, newbie_adjustment, edge_scaling_base,
        require_model_agreement, model_agreement_min_edge, max_decimal_odds,
    ):
        return False

    # Filter 3: Scaled edge threshold
    decimal_odds = implied_prob_to_decimal_odds(market_prob)
    required_edge = required_edge_for_market(
        market_prob,
        edge_scaling_base=edge_scaling_base,
        newbie_adjustment=newbie_adjustment,
    )
    if edge < required_edge:
        logger.debug(
            f"Skipping {fighter_name}: edge {edge:.1%} below scaled "
            f"threshold {required_edge:.1%} at odds {decimal_odds:.2f}"
        )
        return False

    return True


def _filter_rejection_reason(
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
    a_org_tier: object = None,
    b_org_tier: object = None,
    newbie_rule: Optional[NewbieRuleConfig] = None,
    newbie_adjustment: Optional[NewbieAdjustment] = None,
    edge_scaling_base: Optional[float] = None,
    require_model_agreement: Optional[bool] = None,
    model_agreement_min_edge: Optional[float] = None,
    max_decimal_odds: Optional[float] = None,
) -> Optional[dict]:
    """Return ``None`` if the bet passes all filters, otherwise a dict with
    ``reason`` (short label) and ``detail`` (one-liner with numbers)."""
    decimal_odds = implied_prob_to_decimal_odds(market_prob)
    _max_decimal_odds = MAX_DECIMAL_ODDS if max_decimal_odds is None else max_decimal_odds
    _require = REQUIRE_MODEL_AGREEMENT if require_model_agreement is None else require_model_agreement
    _agree_edge = MODEL_AGREEMENT_MIN_EDGE if model_agreement_min_edge is None else model_agreement_min_edge

    adj = newbie_adjustment or newbie_penalty(
        a_num_fights, b_num_fights,
        org_tier_a=a_org_tier, org_tier_b=b_org_tier,
        newbie_rule=newbie_rule,
    )
    if not adj.allowed:
        return {"reason": "Low experience", "detail": adj.reason}

    if blended_prob < MIN_MODEL_PROB:
        return {
            "reason": "Prob too low",
            "detail": f"Blended prob {blended_prob:.0%} below {MIN_MODEL_PROB:.0%} floor",
        }
    if decimal_odds > _max_decimal_odds:
        return {
            "reason": "Odds too long",
            "detail": f"Decimal odds {decimal_odds:.2f} exceed {_max_decimal_odds:.1f} cap",
        }
    if _require:
        if no_odds_prob is None:
            return {"reason": "No-odds unavailable", "detail": "No-odds model probability missing"}
        no_odds_edge = no_odds_prob - market_prob
        if no_odds_edge < _agree_edge:
            return {
                "reason": "No-odds disagrees",
                "detail": f"No-odds edge {no_odds_edge:+.1%}, needs {_agree_edge:+.1%}",
            }

    if LINE_MOVEMENT_FILTER and line_movement is not None and bet_side is not None:
        line_against = (bet_side == "a" and line_movement < -0.02) or \
                       (bet_side == "b" and line_movement > 0.02)
        if line_against:
            is_sharp = line_is_sharp == 1 if line_is_sharp is not None else False
            is_steam = line_steam_move == 1 if line_steam_move is not None else False
            if LINE_SHARP_BLOCK and (is_sharp or is_steam):
                return {
                    "reason": "Sharp money against",
                    "detail": f"Sharp/steam move against bet (movement {line_movement:+.1%})",
                }
            req = required_edge_for_market(market_prob, edge_scaling_base=edge_scaling_base, newbie_adjustment=adj)
            extra_required = req + LINE_AGAINST_EXTRA_EDGE
            if edge < extra_required:
                return {
                    "reason": "Line moved against",
                    "detail": f"Line against (movement {line_movement:+.1%}), edge {edge:.1%} < {extra_required:.1%}",
                }

    req_edge = required_edge_for_market(market_prob, edge_scaling_base=edge_scaling_base, newbie_adjustment=adj)
    if edge < req_edge:
        shortfall = req_edge - edge
        return {
            "reason": "Edge below threshold",
            "detail": f"Edge {edge:.1%} needs +{shortfall:.1%} more to clear {req_edge:.1%} at {decimal_odds:.2f} odds",
        }

    return None


def _build_bet_reason(
    fighter: str,
    model_prob: float,
    blend_prob: float,
    market_prob: float,
    edge: float,
    newbie_reason: str,
) -> str:
    """Build a human-readable reason string for why a bet was placed."""
    parts = [f"Model {model_prob:.0%} vs market {market_prob:.0%}, {edge:.1%} edge"]
    if newbie_reason:
        parts.append(newbie_reason)
    return "; ".join(parts)


def _build_conviction_reason(
    fighter: str,
    model_prob: float,
    no_odds_prob: float,
    market_prob: float,
) -> str:
    """Build a human-readable reason string for a conviction bet."""
    return (
        f"Conviction signal on {fighter}: model {model_prob:.0%}, "
        f"no-odds {no_odds_prob:.0%}, market {market_prob:.0%}, positive EV confirmed"
    )


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
    blend_weight: float = BLEND_WEIGHT,
    near_miss_min_edge: float = 0.0,
    edge_scaling_base: Optional[float] = None,
    max_decimal_odds: Optional[float] = None,
    newbie_rule: Optional[NewbieRuleConfig] = None,
):
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

    Args:
        near_miss_min_edge: If > 0, also collect near-miss bets (pass quality
            filters but edge is between near_miss_min_edge and the required
            threshold). Returns (value_bets_df, near_miss_df) when enabled,
            otherwise returns just value_bets_df.

    Returns DataFrame of value bets, or tuple of (value_bets, near_misses).
    """
    bets = []
    near_misses = []
    skipped = 0

    for _, row in predictions.iterrows():
        model_a = _coerce_probability(row.get("prob_a"))
        model_b = _coerce_probability(row.get("prob_b"))
        market_a = _first_valid_probability(
            row.get("a_market_prob"),
            row.get("a_fair_prob_avg"),
            default=np.nan,
        )
        market_b = _first_valid_probability(
            row.get("b_market_prob"),
            row.get("b_fair_prob_avg"),
            default=np.nan,
        )
        # Skip rows with missing model or market probabilities
        if model_a is None or model_b is None or np.isnan(market_a) or np.isnan(market_b):
            skipped += 1
            continue
        no_odds_a = row.get("no_odds_prob_a")
        no_odds_b = row.get("no_odds_prob_b")

        # Line movement metadata (for sharp money filtering)
        line_movement = row.get("line_movement")
        line_is_sharp = row.get("line_is_sharp")
        line_steam_move = row.get("line_steam_move")
        if isinstance(line_movement, float) and np.isnan(line_movement):
            line_movement = None

        # Fighter experience
        a_fights = _coerce_optional_int(row.get("a_num_fights"))
        b_fights = _coerce_optional_int(row.get("b_num_fights"))
        a_org_tier = _coerce_optional_float(row.get("a_pre_ufc_org_tier_best"))
        b_org_tier = _coerce_optional_float(row.get("b_pre_ufc_org_tier_best"))
        newbie_adjustment = newbie_penalty(
            a_fights,
            b_fights,
            org_tier_a=a_org_tier,
            org_tier_b=b_org_tier,
            newbie_rule=newbie_rule,
        )

        blend_a, blend_b = compute_independent_blend_probs(
            model_a,
            market_a,
            no_odds_a,
            model_b,
            market_b,
            no_odds_b,
            blend_weight,
        )

        # Edge = blended - market
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        # Helper: common filter kwargs for a given side
        def _filter_kwargs(side):
            return dict(
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side=side,
                a_num_fights=a_fights, b_num_fights=b_fights,
                a_org_tier=a_org_tier, b_org_tier=b_org_tier,
                newbie_rule=newbie_rule, newbie_adjustment=newbie_adjustment,
            )

        # Helper: build bet dict for a given side
        def _make_bet(side, fighter, model_p, blend_p, market_p, edge_val):
            bet = {
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter,
                "bet_side": side,
                "model_prob": model_p,
                "blended_prob": blend_p,
                "market_prob": market_p,
                "edge": edge_val,
                "decimal_odds": implied_prob_to_decimal_odds(market_p),
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": max(model_a, model_b),
                "is_newbie_bet": newbie_adjustment.is_newbie_bet,
                "size_multiplier": newbie_adjustment.size_multiplier,
                "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
                "newbie_rule": _newbie_rule_config(newbie_rule).name,
                "reason": _build_bet_reason(
                    fighter, model_p, blend_p, market_p, edge_val,
                    newbie_adjustment.reason,
                ),
            }
            for col in ("token_id_yes", "token_id_no", "market_id",
                        "tick_size", "neg_risk", "volume"):
                if row.get(col) is not None:
                    bet[col] = row[col]
            return bet

        # Evaluate both sides independently and take the best passing side.
        added = False
        passing_candidates = []
        failed_value_candidates = 0
        value_candidates = [
            ("a", row.get("fighter_a", "A"), model_a, blend_a, market_a, edge_a, no_odds_a),
            ("b", row.get("fighter_b", "B"), model_b, blend_b, market_b, edge_b, no_odds_b),
        ]
        value_candidates.sort(key=lambda candidate: candidate[5], reverse=True)

        for side, fighter_name, model_p, blend_p, market_p, edge_val, no_odds_p in value_candidates:
            if edge_val < min_edge:
                continue
            if _passes_filters(
                blend_p, market_p, edge_val, fighter_name, no_odds_p,
                edge_scaling_base=edge_scaling_base,
                max_decimal_odds=max_decimal_odds,
                **_filter_kwargs(side),
            ):
                passing_candidates.append((side, fighter_name, model_p, blend_p, market_p, edge_val))
            else:
                failed_value_candidates += 1

        if passing_candidates:
            side, fighter_name, model_p, blend_p, market_p, edge_val = passing_candidates[0]
            bets.append(_make_bet(side, fighter_name, model_p, blend_p, market_p, edge_val))
            added = True
        elif failed_value_candidates:
            skipped += 1

        # Near-miss collection: if not already a value bet, check if either side
        # passes quality filters with edge in the near-miss range
        if not added and near_miss_min_edge > 0:
            # Pick the side with the best edge for near-miss consideration
            candidates = []
            if edge_a >= near_miss_min_edge:
                candidates.append(("a", row.get("fighter_a", "A"), model_a, blend_a, market_a, edge_a, no_odds_a))
            if edge_b >= near_miss_min_edge:
                candidates.append(("b", row.get("fighter_b", "B"), model_b, blend_b, market_b, edge_b, no_odds_b))

            # Sort by edge descending, pick best
            candidates.sort(key=lambda c: c[5], reverse=True)

            for side, fighter_name, model_p, blend_p, market_p, edge_val, no_odds_p in candidates:
                # Must pass all quality filters (everything except edge threshold)
                if not _passes_quality_filters(
                    blend_p, market_p, edge_val, fighter_name, no_odds_p,
                    edge_scaling_base=edge_scaling_base,
                    max_decimal_odds=max_decimal_odds,
                    **_filter_kwargs(side),
                ):
                    continue

                # Edge must be below the scaled required edge (otherwise it would
                # have been caught as a value bet above)
                decimal_odds = implied_prob_to_decimal_odds(market_p)
                required_edge = required_edge_for_market(
                    market_p,
                    edge_scaling_base=edge_scaling_base,
                    newbie_adjustment=newbie_adjustment,
                )
                if edge_val >= required_edge:
                    continue  # Should have been a value bet — skip

                near_misses.append(_make_bet(side, fighter_name, model_p, blend_p, market_p, edge_val))
                break  # Only one near-miss per fight

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

    if near_miss_min_edge > 0:
        nm_result = pd.DataFrame(near_misses)
        if not nm_result.empty:
            nm_result = nm_result.sort_values("edge", ascending=False).reset_index(drop=True)
            logger.info(
                f"Found {len(nm_result)} near-miss candidates "
                f"(edge {near_miss_min_edge:.1%}–threshold). "
                f"Avg edge: {nm_result['edge'].mean():.1%}"
            )
        else:
            logger.info("No near-miss candidates found")
        return result, nm_result

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

    # Bonus: +1% bankroll for every 5% model prob above the threshold
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
    *,
    require_positive_ev: bool = True,
) -> pd.DataFrame:
    """
    Identify conviction bets — fighters that both models agree will win.

    Conviction bets still require a positive expected value at the current market.
    They require dual model agreement:
      1. XGBoost model prob >= CONVICTION_MIN_MODEL_PROB (65%)
      2. No-odds model prob >= CONVICTION_MIN_NO_ODDS_PROB (50%)
    """
    bets = []
    skipped = 0

    for _, row in predictions.iterrows():
        model_a = _coerce_probability(row.get("prob_a"))
        model_b = _coerce_probability(row.get("prob_b"))
        market_a = _first_valid_probability(
            row.get("a_market_prob"),
            row.get("a_fair_prob_avg"),
            default=np.nan,
        )
        market_b = _first_valid_probability(
            row.get("b_market_prob"),
            row.get("b_fair_prob_avg"),
            default=np.nan,
        )
        # Skip rows with missing model or market probabilities
        if model_a is None or model_b is None or np.isnan(market_a) or np.isnan(market_b):
            skipped += 1
            continue
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

            decimal_odds = implied_prob_to_decimal_odds(market_p)
            if require_positive_ev and calculate_expected_value(model_p, decimal_odds) <= 0:
                logger.debug(
                    f"Conviction skip {fighter_name}: non-positive EV at market prob {market_p:.1%}"
                )
                skipped += 1
                continue

            # All gates passed — this is a conviction bet
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
                "reason": _build_conviction_reason(
                    fighter_name,
                    model_p,
                    no_odds_p,
                    market_p,
                ),
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
        logger.info("No conviction bets found (dual model agreement not met for any fight)")

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
