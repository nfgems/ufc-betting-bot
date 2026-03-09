"""
Line movement tracker — monitors betting odds over time to detect sharp money.

Tracks odds from both The Odds API and Polymarket, storing snapshots
at regular intervals. Sharp line movement (big moves in short time)
is one of the strongest predictive signals in sports betting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.odds_client import OddsClient
from src.polymarket.markets import get_ufc_fight_markets
from src.config import RAW_DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

LINE_HISTORY_DIR = RAW_DATA_DIR / "line_history"
LINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def snapshot_odds() -> pd.DataFrame:
    """
    Take a snapshot of current odds from The Odds API.
    Saves to line history and returns the data.
    """
    client = OddsClient()
    try:
        odds = client.get_live_odds()
        df = client.odds_to_dataframe(odds)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Add timestamp
    df["snapshot_time"] = datetime.now().isoformat()

    # Save snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LINE_HISTORY_DIR / f"odds_{timestamp}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved odds snapshot: {len(df)} records to {path}")

    return df


def snapshot_polymarket_prices() -> pd.DataFrame:
    """
    Take a snapshot of current Polymarket UFC market prices.
    """
    try:
        markets = get_ufc_fight_markets()
    except Exception as e:
        logger.error(f"Failed to fetch Polymarket prices: {e}")
        return pd.DataFrame()

    if markets.empty:
        return markets

    markets["snapshot_time"] = datetime.now().isoformat()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LINE_HISTORY_DIR / f"polymarket_{timestamp}.csv"
    markets.to_csv(path, index=False)
    logger.info(f"Saved Polymarket snapshot: {len(markets)} markets to {path}")

    return markets


def load_line_history(fighter_a: str, fighter_b: str) -> pd.DataFrame:
    """
    Load all historical odds snapshots for a specific fight.

    Returns DataFrame with columns:
        snapshot_time, bookmaker, a_odds, b_odds, a_implied_prob, b_implied_prob
    """
    all_snapshots = []
    fa = fighter_a.lower()
    fb = fighter_b.lower()

    for csv_path in sorted(LINE_HISTORY_DIR.glob("odds_*.csv")):
        try:
            df = pd.read_csv(csv_path)
            # Filter to this fight
            mask = (
                (df["fighter_a"].str.lower().str.contains(fa, na=False) &
                 df["fighter_b"].str.lower().str.contains(fb, na=False)) |
                (df["fighter_a"].str.lower().str.contains(fb, na=False) &
                 df["fighter_b"].str.lower().str.contains(fa, na=False))
            )
            matched = df[mask]
            if not matched.empty:
                all_snapshots.append(matched)
        except Exception:
            continue

    if not all_snapshots:
        return pd.DataFrame()

    history = pd.concat(all_snapshots, ignore_index=True)
    history["snapshot_time"] = pd.to_datetime(history["snapshot_time"])
    history = history.sort_values("snapshot_time")

    return history


def analyze_line_movement(fighter_a: str, fighter_b: str) -> dict:
    """
    Analyze line movement for a specific fight.

    Returns dict with:
        - opening_prob_a: opening implied probability for fighter A
        - current_prob_a: current implied probability
        - movement: total probability shift (positive = moved toward A)
        - max_movement: largest single-snapshot move
        - is_sharp_move: True if line moved sharply (>5% in <24 hours)
        - direction: "toward_a", "toward_b", or "stable"
        - num_snapshots: number of data points
        - steam_move: True if sharp, sudden move detected (indicates sharp money)
    """
    history = load_line_history(fighter_a, fighter_b)

    if history.empty or len(history) < 2:
        return {
            "opening_prob_a": None,
            "current_prob_a": None,
            "movement": 0.0,
            "is_sharp_move": False,
            "direction": "unknown",
            "num_snapshots": len(history),
            "steam_move": False,
        }

    # Get consensus (average across bookmakers) at each snapshot
    consensus = history.groupby("snapshot_time").agg(
        a_prob=("a_fair_prob", "mean"),
        b_prob=("b_fair_prob", "mean"),
    ).reset_index()

    opening_a = consensus["a_prob"].iloc[0]
    current_a = consensus["a_prob"].iloc[-1]
    movement = current_a - opening_a

    # Detect sharp moves (>3% shift between consecutive snapshots)
    consensus["a_shift"] = consensus["a_prob"].diff()
    max_shift = consensus["a_shift"].abs().max()

    # Steam move: >5% total movement or >3% in a single snapshot
    is_sharp = abs(movement) > 0.05 or max_shift > 0.03

    # Detect steam move pattern: rapid movement in one direction
    recent_shifts = consensus["a_shift"].tail(3)
    steam_move = (
        all(s > 0.01 for s in recent_shifts.dropna()) or
        all(s < -0.01 for s in recent_shifts.dropna())
    )

    if movement > 0.02:
        direction = "toward_a"
    elif movement < -0.02:
        direction = "toward_b"
    else:
        direction = "stable"

    result = {
        "opening_prob_a": float(opening_a),
        "current_prob_a": float(current_a),
        "opening_prob_b": float(1 - opening_a),
        "current_prob_b": float(1 - current_a),
        "movement": float(movement),
        "abs_movement": float(abs(movement)),
        "max_single_shift": float(max_shift),
        "is_sharp_move": is_sharp,
        "steam_move": steam_move,
        "direction": direction,
        "num_snapshots": len(consensus),
        "hours_tracked": (
            (consensus["snapshot_time"].iloc[-1] - consensus["snapshot_time"].iloc[0])
            .total_seconds() / 3600
        ),
    }

    if is_sharp:
        logger.info(
            f"SHARP LINE MOVE: {fighter_a} vs {fighter_b} | "
            f"Opened {opening_a:.1%} -> Now {current_a:.1%} "
            f"({movement:+.1%} {direction})"
        )
    if steam_move:
        logger.warning(
            f"STEAM MOVE DETECTED: {fighter_a} vs {fighter_b} | "
            f"Consistent movement {direction} — likely sharp money"
        )

    return result


def detect_injury_or_cancellation(
    fighter_a: str,
    fighter_b: str,
    current_odds: Optional[dict] = None,
) -> dict:
    """
    Detect if a fight has likely been affected by injury, cancellation, or
    other fight-breaking news based on extreme odds movement.

    Checks two signals:
    1. Extreme line movement (>15% shift from opening) — indicates sudden news
    2. One side near zero (<5¢ on Polymarket) — fight is essentially off

    Args:
        fighter_a, fighter_b: fighter names
        current_odds: optional dict with current market prices (a_prob, b_prob)

    Returns dict with:
        - suspected: True if injury/cancellation detected
        - reason: description of what was detected
        - severity: "warning" or "block" (block = do not bet)
    """
    from src.config import INJURY_MOVE_THRESHOLD, INJURY_PRICE_FLOOR

    result = {
        "suspected": False,
        "reason": "",
        "severity": "ok",
        "details": {},
    }

    # Check 1: Extreme line movement from tracked history
    analysis = analyze_line_movement(fighter_a, fighter_b)
    abs_move = abs(analysis.get("movement", 0))

    if abs_move >= INJURY_MOVE_THRESHOLD:
        direction = analysis.get("direction", "unknown")
        moved_away_from = fighter_b if direction == "toward_a" else fighter_a
        result["suspected"] = True
        result["severity"] = "block"
        result["reason"] = (
            f"Extreme line movement ({abs_move:.0%}) away from {moved_away_from} — "
            f"possible injury/cancellation"
        )
        result["details"] = analysis
        logger.warning(
            f"INJURY ALERT: {fighter_a} vs {fighter_b} — "
            f"{abs_move:.0%} line shift detected. {result['reason']}"
        )
        return result

    # Check 2: One side near zero in current market prices
    if current_odds:
        a_prob = current_odds.get("a_prob", 0.5)
        b_prob = current_odds.get("b_prob", 0.5)

        if a_prob < INJURY_PRICE_FLOOR:
            result["suspected"] = True
            result["severity"] = "block"
            result["reason"] = (
                f"{fighter_a} price at {a_prob:.0%} (below {INJURY_PRICE_FLOOR:.0%}) — "
                f"fight likely cancelled or {fighter_a} withdrawn"
            )
            logger.warning(f"INJURY ALERT: {result['reason']}")
            return result

        if b_prob < INJURY_PRICE_FLOOR:
            result["suspected"] = True
            result["severity"] = "block"
            result["reason"] = (
                f"{fighter_b} price at {b_prob:.0%} (below {INJURY_PRICE_FLOOR:.0%}) — "
                f"fight likely cancelled or {fighter_b} withdrawn"
            )
            logger.warning(f"INJURY ALERT: {result['reason']}")
            return result

    # Check 3: Steam move (softer signal — warning, don't block)
    if analysis.get("steam_move"):
        result["suspected"] = True
        result["severity"] = "warning"
        result["reason"] = (
            f"Steam move detected ({analysis.get('movement', 0):+.1%} "
            f"{analysis.get('direction', 'unknown')}) — monitor closely"
        )
        result["details"] = analysis

    return result


def get_line_movement_features(fighter_a: str, fighter_b: str) -> dict:
    """
    Get line movement features for the prediction model.

    These features capture where the "smart money" is going,
    which is one of the strongest signals in sports betting.
    """
    analysis = analyze_line_movement(fighter_a, fighter_b)

    return {
        "line_movement": analysis.get("movement", 0.0),
        "line_abs_movement": analysis.get("abs_movement", 0.0),
        "line_is_sharp": 1 if analysis.get("is_sharp_move") else 0,
        "line_steam_move": 1 if analysis.get("steam_move") else 0,
        "line_direction_toward_a": 1 if analysis.get("direction") == "toward_a" else 0,
        "line_direction_toward_b": 1 if analysis.get("direction") == "toward_b" else 0,
    }


def run_line_tracking_pass() -> dict:
    """Run a complete line tracking pass — snapshot odds + Polymarket + analyze."""
    logger.info("Running line tracking pass...")

    odds_df = snapshot_odds()
    poly_df = snapshot_polymarket_prices()

    # Analyze movement for all tracked fights
    analyses = {}
    if not odds_df.empty:
        fights = odds_df.groupby(["fighter_a", "fighter_b"]).first().reset_index()
        for _, fight in fights.iterrows():
            key = f"{fight['fighter_a']} vs {fight['fighter_b']}"
            analyses[key] = analyze_line_movement(fight["fighter_a"], fight["fighter_b"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "odds_records": len(odds_df),
        "polymarket_markets": len(poly_df),
        "fights_analyzed": len(analyses),
        "sharp_moves": sum(1 for a in analyses.values() if a.get("is_sharp_move")),
        "steam_moves": sum(1 for a in analyses.values() if a.get("steam_move")),
        "analyses": analyses,
    }

    logger.info(
        f"Line tracking: {len(analyses)} fights, "
        f"{summary['sharp_moves']} sharp moves, "
        f"{summary['steam_moves']} steam moves"
    )

    return summary
