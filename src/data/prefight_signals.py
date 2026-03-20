"""
Pre-fight signals aggregator — collects all non-statistical signals
that affect fight outcomes and converts them to model features.

Signals collected:
1. Missed weight → penalize the fighter who missed
2. Short-notice replacement → penalize the replacement fighter
3. Referee assignment → adjust for ref tendencies
4. Line movement → capture sharp money direction
5. Fighter activity → penalize long layoffs
6. Camp changes → detect via news (manual input for now)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.live_monitor import (
    check_missed_weight,
    detect_short_notice,
    get_referee_features,
    load_latest_snapshot,
)
from src.data.line_tracker import get_line_movement_features
from src.config import LOGS_DIR

logger = logging.getLogger(__name__)

SIGNALS_DIR = LOGS_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Signal impact weights (how much each signal adjusts win probability)
# Based on historical analysis of UFC fight outcomes
# ---------------------------------------------------------------------------

SIGNAL_ADJUSTMENTS = {
    # Missed weight: fighter who missed loses ~8% win probability on average
    # Source: UFC data shows fighters who miss weight win at ~47% vs ~53% expected
    "missed_weight": -0.08,
    "missed_weight_per_pound": -0.02,  # Additional penalty per pound over

    # Short notice: replacement fighters win at ~38% vs ~50% baseline
    "short_notice_14_days": -0.12,
    "short_notice_7_days": -0.18,
    "short_notice_3_days": -0.22,

    # Layoff: fighters returning after 12+ months win at ~44%
    "layoff_12_months": -0.06,
    "layoff_18_months": -0.10,
    "layoff_24_months": -0.14,

    # Debut/first UFC fight: ~42% win rate historically
    "ufc_debut": -0.08,
}


def collect_prefight_signals(
    fighter_a: str,
    fighter_b: str,
    event_title: str = "",
    referee: str = "",
    a_days_since_fight: Optional[int] = None,
    b_days_since_fight: Optional[int] = None,
    a_is_short_notice: bool = False,
    b_is_short_notice: bool = False,
    a_days_notice: int = 60,
    b_days_notice: int = 60,
    a_missed_weight: bool = False,
    b_missed_weight: bool = False,
    a_weight_over: float = 0.0,
    b_weight_over: float = 0.0,
    a_is_debut: bool = False,
    b_is_debut: bool = False,
) -> dict:
    """
    Collect all pre-fight signals and compute probability adjustments.

    Returns dict with:
        - Individual signal flags and values
        - Total probability adjustment for each fighter
        - Feature dict ready for model input
    """
    signals = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "event": event_title,
        "collected_at": datetime.now().isoformat(),
        "adjustments": {"a": 0.0, "b": 0.0},
        "flags": [],
    }

    # --- Missed weight ---
    if a_missed_weight:
        adj = SIGNAL_ADJUSTMENTS["missed_weight"]
        adj += SIGNAL_ADJUSTMENTS["missed_weight_per_pound"] * a_weight_over
        signals["adjustments"]["a"] += adj
        signals["flags"].append(f"{fighter_a} missed weight by {a_weight_over:.1f} lbs (adj: {adj:+.1%})")

    if b_missed_weight:
        adj = SIGNAL_ADJUSTMENTS["missed_weight"]
        adj += SIGNAL_ADJUSTMENTS["missed_weight_per_pound"] * b_weight_over
        signals["adjustments"]["b"] += adj
        signals["flags"].append(f"{fighter_b} missed weight by {b_weight_over:.1f} lbs (adj: {adj:+.1%})")

    # --- Short notice ---
    if a_is_short_notice:
        if a_days_notice <= 3:
            adj = SIGNAL_ADJUSTMENTS["short_notice_3_days"]
        elif a_days_notice <= 7:
            adj = SIGNAL_ADJUSTMENTS["short_notice_7_days"]
        else:
            adj = SIGNAL_ADJUSTMENTS["short_notice_14_days"]
        signals["adjustments"]["a"] += adj
        signals["flags"].append(f"{fighter_a} on {a_days_notice}-day notice (adj: {adj:+.1%})")

    if b_is_short_notice:
        if b_days_notice <= 3:
            adj = SIGNAL_ADJUSTMENTS["short_notice_3_days"]
        elif b_days_notice <= 7:
            adj = SIGNAL_ADJUSTMENTS["short_notice_7_days"]
        else:
            adj = SIGNAL_ADJUSTMENTS["short_notice_14_days"]
        signals["adjustments"]["b"] += adj
        signals["flags"].append(f"{fighter_b} on {b_days_notice}-day notice (adj: {adj:+.1%})")

    # --- Layoff ---
    if a_days_since_fight and a_days_since_fight > 365:
        months = a_days_since_fight / 30.44
        if months >= 24:
            adj = SIGNAL_ADJUSTMENTS["layoff_24_months"]
        elif months >= 18:
            adj = SIGNAL_ADJUSTMENTS["layoff_18_months"]
        else:
            adj = SIGNAL_ADJUSTMENTS["layoff_12_months"]
        signals["adjustments"]["a"] += adj
        signals["flags"].append(f"{fighter_a} returning after {months:.0f} months (adj: {adj:+.1%})")

    if b_days_since_fight and b_days_since_fight > 365:
        months = b_days_since_fight / 30.44
        if months >= 24:
            adj = SIGNAL_ADJUSTMENTS["layoff_24_months"]
        elif months >= 18:
            adj = SIGNAL_ADJUSTMENTS["layoff_18_months"]
        else:
            adj = SIGNAL_ADJUSTMENTS["layoff_12_months"]
        signals["adjustments"]["b"] += adj
        signals["flags"].append(f"{fighter_b} returning after {months:.0f} months (adj: {adj:+.1%})")

    # --- UFC debut ---
    if a_is_debut:
        adj = SIGNAL_ADJUSTMENTS["ufc_debut"]
        signals["adjustments"]["a"] += adj
        signals["flags"].append(f"{fighter_a} making UFC debut (adj: {adj:+.1%})")

    if b_is_debut:
        adj = SIGNAL_ADJUSTMENTS["ufc_debut"]
        signals["adjustments"]["b"] += adj
        signals["flags"].append(f"{fighter_b} making UFC debut (adj: {adj:+.1%})")

    # --- Referee ---
    if referee:
        ref_features = get_referee_features(referee)
        signals["referee"] = referee
        signals["ref_features"] = ref_features

    # --- Line movement ---
    line_features = get_line_movement_features(fighter_a, fighter_b)
    signals["line_features"] = line_features

    # Log signals
    if signals["flags"]:
        logger.info(f"\nPre-fight signals for {fighter_a} vs {fighter_b}:")
        for flag in signals["flags"]:
            logger.info(f"  * {flag}")
        logger.info(
            f"  Net adjustment: A {signals['adjustments']['a']:+.1%}, "
            f"B {signals['adjustments']['b']:+.1%}"
        )

    return signals


def signals_to_features(signals: dict) -> dict:
    """
    Convert collected signals to a flat feature dict for the model.

    Returns dict of feature_name -> value that can be merged into
    the main feature matrix before prediction.
    """
    features = {}

    # Missed weight features — use the dedicated flag text, not the combined adjustment
    flags_text = " ".join(signals.get("flags", []))
    a_name = signals.get("fighter_a", "")
    b_name = signals.get("fighter_b", "")
    features["a_missed_weight"] = 1 if (a_name and f"{a_name} missed weight" in flags_text) else 0
    features["b_missed_weight"] = 1 if (b_name and f"{b_name} missed weight" in flags_text) else 0

    # Probability adjustments
    features["a_signal_adjustment"] = signals["adjustments"]["a"]
    features["b_signal_adjustment"] = signals["adjustments"]["b"]
    features["diff_signal_adjustment"] = (
        signals["adjustments"]["a"] - signals["adjustments"]["b"]
    )

    # Referee features
    ref = signals.get("ref_features", {})
    features["ref_standup_tendency"] = ref.get("ref_standup_tendency", 0.5)
    features["ref_stoppage_tendency"] = ref.get("ref_stoppage_tendency", 0.5)

    # Line movement features
    line = signals.get("line_features", {})
    features.update(line)

    # Count of negative signals per fighter
    features["a_num_flags"] = sum(
        1 for f in signals.get("flags", [])
        if signals["fighter_a"].lower() in f.lower()
    )
    features["b_num_flags"] = sum(
        1 for f in signals.get("flags", [])
        if signals["fighter_b"].lower() in f.lower()
    )

    return features


def adjust_prediction(
    model_prob_a: float,
    signals: dict,
    signal_weight: float = 0.5,
) -> tuple[float, float]:
    """
    Adjust model probability using pre-fight signals.

    Blends the model's statistical prediction with signal-based adjustments.
    signal_weight controls how much signals affect the final probability
    (0 = ignore signals, 1 = full signal adjustment).

    Returns (adjusted_prob_a, adjusted_prob_b).
    """
    adj_a = signals["adjustments"]["a"]
    adj_b = signals["adjustments"]["b"]

    # Net adjustment relative to fighter A
    net_adj = (adj_a - adj_b) * signal_weight

    # Apply adjustment
    adjusted_a = model_prob_a + net_adj
    adjusted_a = max(0.01, min(0.99, adjusted_a))  # Clamp to 1-99%
    adjusted_b = 1.0 - adjusted_a

    if abs(net_adj) > 0.01:
        logger.info(
            f"Adjusted prediction: {model_prob_a:.1%} -> {adjusted_a:.1%} "
            f"(signal adj: {net_adj:+.1%})"
        )

    return adjusted_a, adjusted_b
