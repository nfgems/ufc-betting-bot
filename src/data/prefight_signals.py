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


def _short_notice_flag(
    fighter: str,
    days_until_event_at_detection: Optional[int],
) -> str:
    flag = f"{fighter} is a short-notice replacement"
    if days_until_event_at_detection is None:
        return flag
    if isinstance(days_until_event_at_detection, bool):
        return flag
    try:
        countdown = max(0, int(days_until_event_at_detection))
    except (TypeError, ValueError, OverflowError):
        return flag
    unit = "day" if countdown == 1 else "days"
    return f"{flag} (late-detected with {countdown} {unit} until event)"


def collect_prefight_signals(
    fighter_a: str,
    fighter_b: str,
    event_title: str = "",
    referee: str = "",
    a_days_since_fight: Optional[int] = None,
    b_days_since_fight: Optional[int] = None,
    a_is_short_notice: bool = False,
    b_is_short_notice: bool = False,
    a_days_until_event_at_detection: Optional[int] = None,
    b_days_until_event_at_detection: Optional[int] = None,
    a_missed_weight: bool = False,
    b_missed_weight: bool = False,
    a_weight_over: float = 0.0,
    b_weight_over: float = 0.0,
    a_is_debut: bool = False,
    b_is_debut: bool = False,
) -> dict:
    """
    Collect all pre-fight signals and log them as observational flags.

    Returns dict with:
        - Individual signal flags and values (for logging/auditing only)
        - Feature dict ready for model input
    Note: signals no longer apply hardcoded probability adjustments.
    """
    signals = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "event": event_title,
        "collected_at": datetime.now().isoformat(),
        "flags": [],
    }

    # --- Missed weight ---
    if a_missed_weight:
        signals["flags"].append(f"{fighter_a} missed weight by {a_weight_over:.1f} lbs")

    if b_missed_weight:
        signals["flags"].append(f"{fighter_b} missed weight by {b_weight_over:.1f} lbs")

    # --- Short notice ---
    if a_is_short_notice:
        signals["flags"].append(
            _short_notice_flag(
                fighter_a,
                a_days_until_event_at_detection,
            )
        )

    if b_is_short_notice:
        signals["flags"].append(
            _short_notice_flag(
                fighter_b,
                b_days_until_event_at_detection,
            )
        )

    # --- Layoff ---
    if a_days_since_fight and a_days_since_fight > 365:
        months = a_days_since_fight / 30.44
        signals["flags"].append(f"{fighter_a} returning after {months:.0f} months")

    if b_days_since_fight and b_days_since_fight > 365:
        months = b_days_since_fight / 30.44
        signals["flags"].append(f"{fighter_b} returning after {months:.0f} months")

    # --- UFC debut ---
    if a_is_debut:
        signals["flags"].append(f"{fighter_a} making UFC debut")

    if b_is_debut:
        signals["flags"].append(f"{fighter_b} making UFC debut")

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


