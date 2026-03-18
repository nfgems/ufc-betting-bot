"""
Line movement tracker - monitors betting odds over time to detect sharp money.

The collector owns line-history creation. Feature builders only read persisted
history and never fabricate an opening line on first observation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR
from src.data.line_movement import (
    analysis_to_line_movement_features,
    compute_line_movement_analysis,
)
from src.data.name_utils import normalize_person_name
from src.data.odds_client import OddsClient
from src.polymarket.markets import get_ufc_fight_markets

logger = logging.getLogger(__name__)

LINE_HISTORY_DIR = RAW_DATA_DIR / "line_history"
LINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
OPENING_LINES_PATH = LINE_HISTORY_DIR / "opening_lines.json"

_SWAPPABLE_COLUMNS = [
    ("fighter_a", "fighter_b"),
    ("fighter_a_norm", "fighter_b_norm"),
    ("a_odds", "b_odds"),
    ("a_implied_prob", "b_implied_prob"),
    ("a_fair_prob", "b_fair_prob"),
]


def _normalize_fighter_name(name: str) -> str:
    """Normalize a fighter name for exact snapshot matching."""
    return normalize_person_name(name)


def _pair_key(fighter_a: str, fighter_b: str) -> str:
    normalized = sorted([_normalize_fighter_name(fighter_a), _normalize_fighter_name(fighter_b)])
    return "|".join(normalized)


def _parse_datetime_like(value: object) -> Optional[pd.Timestamp]:
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _as_of_cutoff_timestamp(value: object) -> Optional[pd.Timestamp]:
    cutoff = _parse_datetime_like(value)
    if cutoff is None:
        return None
    text = str(value or "").strip()
    if "T" not in text and " " not in text:
        cutoff = cutoff + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return cutoff


def _fight_key(event_id: object, commence_time: object, fighter_a: str, fighter_b: str) -> str:
    event_id_text = str(event_id or "").strip()
    if event_id_text and event_id_text.lower() != "nan":
        return f"event::{event_id_text}"

    commence = _parse_datetime_like(commence_time)
    commence_text = commence.isoformat() if commence is not None else "unknown"
    return f"{commence_text}::{_pair_key(fighter_a, fighter_b)}"


def _prepare_snapshot_frame(df: pd.DataFrame, *, snapshot_time: Optional[str] = None) -> pd.DataFrame:
    prepared = df.copy()
    if prepared.empty:
        return prepared

    if snapshot_time is not None:
        prepared["snapshot_time"] = snapshot_time
    elif "snapshot_time" not in prepared.columns:
        prepared["snapshot_time"] = datetime.now().isoformat()

    for column in ["event_id", "commence_time"]:
        if column not in prepared.columns:
            prepared[column] = ""

    prepared["fighter_a_norm"] = prepared["fighter_a"].map(_normalize_fighter_name)
    prepared["fighter_b_norm"] = prepared["fighter_b"].map(_normalize_fighter_name)
    prepared["pair_key"] = prepared.apply(
        lambda row: _pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    prepared["fight_key"] = prepared.apply(
        lambda row: _fight_key(
            row.get("event_id", ""),
            row.get("commence_time", ""),
            row.get("fighter_a", ""),
            row.get("fighter_b", ""),
        ),
        axis=1,
    )
    return prepared


def _load_opening_lines() -> dict:
    if not OPENING_LINES_PATH.exists():
        return {}
    try:
        data = json.loads(OPENING_LINES_PATH.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_opening_lines(data: dict) -> None:
    OPENING_LINES_PATH.write_text(json.dumps(data, indent=2))


def load_opening_lines() -> dict:
    """Expose explicit opening-line records for health checks/tests."""
    return _load_opening_lines()


def _record_opening_lines(snapshot_df: pd.DataFrame) -> int:
    if snapshot_df.empty:
        return 0

    opening_lines = _load_opening_lines()
    recorded = 0
    ordered = snapshot_df.copy()
    ordered["snapshot_time"] = pd.to_datetime(ordered["snapshot_time"], errors="coerce")
    ordered = ordered.sort_values("snapshot_time")

    for _, row in ordered.iterrows():
        fight_key = row.get("fight_key", "")
        if not fight_key or fight_key in opening_lines:
            continue

        a_prob = pd.to_numeric(pd.Series([row.get("a_fair_prob")]), errors="coerce").iloc[0]
        b_prob = pd.to_numeric(pd.Series([row.get("b_fair_prob")]), errors="coerce").iloc[0]
        opening_lines[fight_key] = {
            "fight_key": fight_key,
            "event_id": str(row.get("event_id", "") or ""),
            "commence_time": str(row.get("commence_time", "") or ""),
            "fighter_a": str(row.get("fighter_a", "") or ""),
            "fighter_b": str(row.get("fighter_b", "") or ""),
            "fighter_a_norm": str(row.get("fighter_a_norm", "") or ""),
            "fighter_b_norm": str(row.get("fighter_b_norm", "") or ""),
            "pair_key": str(row.get("pair_key", "") or ""),
            "opening_prob_a": None if pd.isna(a_prob) else float(a_prob),
            "opening_prob_b": None if pd.isna(b_prob) else float(b_prob),
            "recorded_at": (
                row["snapshot_time"].isoformat()
                if isinstance(row["snapshot_time"], pd.Timestamp) and not pd.isna(row["snapshot_time"])
                else str(row.get("snapshot_time", "") or "")
            ),
        }
        recorded += 1

    if recorded:
        _save_opening_lines(opening_lines)
        logger.info("Recorded %s explicit opening lines", recorded)

    return recorded


def save_odds_snapshot(df: pd.DataFrame, *, snapshot_time: Optional[str] = None) -> pd.DataFrame:
    """
    Persist an odds snapshot and update explicit opening-line records.

    This helper lets tests and scheduled collectors share the same write path.
    """
    prepared = _prepare_snapshot_frame(df, snapshot_time=snapshot_time)
    if prepared.empty:
        return prepared

    timestamp_source = snapshot_time or str(prepared["snapshot_time"].iloc[0])
    timestamp = pd.to_datetime(timestamp_source, errors="coerce")
    if pd.isna(timestamp):
        filename_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        filename_stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    path = LINE_HISTORY_DIR / f"odds_{filename_stamp}.csv"
    prepared.to_csv(path, index=False)
    opening_lines_recorded = _record_opening_lines(prepared)
    logger.info(
        "Saved odds snapshot: %s records to %s (opening lines recorded: %s)",
        len(prepared),
        path,
        opening_lines_recorded,
    )
    return prepared


def snapshot_odds() -> pd.DataFrame:
    """
    Take a snapshot of current odds from The Odds API.
    Saves to line history and returns the prepared data.
    """
    client = OddsClient()
    try:
        odds = client.get_live_odds()
        df = client.odds_to_dataframe(odds)
    except Exception as exc:
        logger.error("Failed to fetch odds: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    return save_odds_snapshot(df, snapshot_time=datetime.now().isoformat())


def snapshot_polymarket_prices() -> pd.DataFrame:
    """
    Take a snapshot of current Polymarket UFC market prices.
    """
    try:
        markets = get_ufc_fight_markets()
    except Exception as exc:
        logger.error("Failed to fetch Polymarket prices: %s", exc)
        return pd.DataFrame()

    if markets.empty:
        return markets

    markets["snapshot_time"] = datetime.now().isoformat()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LINE_HISTORY_DIR / f"polymarket_{timestamp}.csv"
    markets.to_csv(path, index=False)
    logger.info("Saved Polymarket snapshot: %s markets to %s", len(markets), path)

    return markets


def _reorient_history(history: pd.DataFrame, fighter_a: str) -> pd.DataFrame:
    requested_a = _normalize_fighter_name(fighter_a)
    reoriented = history.copy()
    reverse_mask = reoriented["fighter_a_norm"] != requested_a
    if not reverse_mask.any():
        return reoriented

    for left, right in _SWAPPABLE_COLUMNS:
        if left not in reoriented.columns or right not in reoriented.columns:
            continue
        left_values = reoriented.loc[reverse_mask, left].copy()
        right_values = reoriented.loc[reverse_mask, right].copy()
        reoriented.loc[reverse_mask, left] = right_values
        reoriented.loc[reverse_mask, right] = left_values

    return reoriented


def _select_latest_fight_key(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "fight_key" not in history.columns or history["fight_key"].nunique() <= 1:
        return history

    scored = history.copy()
    scored["commence_time_parsed"] = pd.to_datetime(scored["commence_time"], errors="coerce", utc=True)
    scored["snapshot_time"] = pd.to_datetime(scored["snapshot_time"], errors="coerce", utc=True)

    summaries = (
        scored.groupby("fight_key", dropna=False)
        .agg(
            latest_commence=("commence_time_parsed", "max"),
            latest_snapshot=("snapshot_time", "max"),
        )
        .reset_index()
    )
    summaries["sort_commence"] = summaries["latest_commence"].where(
        summaries["latest_commence"].notna(),
        pd.Timestamp.min,
    )
    summaries["sort_snapshot"] = summaries["latest_snapshot"].where(
        summaries["latest_snapshot"].notna(),
        pd.Timestamp.min,
    )
    selected_key = summaries.sort_values(
        ["sort_commence", "sort_snapshot", "fight_key"],
        ascending=[False, False, False],
    ).iloc[0]["fight_key"]
    return history[history["fight_key"] == selected_key].copy()


def load_line_history(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all historical odds snapshots for a specific fight.

    Returns DataFrame with columns:
        snapshot_time, bookmaker, a_odds, b_odds, a_implied_prob, b_implied_prob
    """
    all_snapshots = []
    pair_key = _pair_key(fighter_a, fighter_b)
    requested_event_id = str(event_id or "")
    requested_commence = _parse_datetime_like(commence_time)
    as_of_cutoff = _as_of_cutoff_timestamp(as_of_date)

    for csv_path in sorted(LINE_HISTORY_DIR.glob("odds_*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "fighter_a" not in df.columns or "fighter_b" not in df.columns:
            continue

        prepared = _prepare_snapshot_frame(df)
        matched = prepared[prepared["pair_key"] == pair_key].copy()
        if matched.empty:
            continue

        if requested_event_id:
            matched = matched[matched["event_id"].astype(str) == requested_event_id]
            if matched.empty:
                continue

        if requested_commence is not None:
            matched["commence_time_parsed"] = pd.to_datetime(
                matched["commence_time"],
                errors="coerce",
                utc=True,
            )
            matched = matched[matched["commence_time_parsed"] == requested_commence]
            if matched.empty:
                continue

        if as_of_cutoff is not None:
            matched["snapshot_time_parsed"] = pd.to_datetime(
                matched["snapshot_time"],
                errors="coerce",
                utc=True,
            )
            matched = matched[
                matched["snapshot_time_parsed"].notna()
                & (matched["snapshot_time_parsed"] <= as_of_cutoff)
            ]
            if matched.empty:
                continue

        all_snapshots.append(matched)

    if not all_snapshots:
        return pd.DataFrame()

    history = pd.concat(all_snapshots, ignore_index=True)
    if not requested_event_id:
        history = _select_latest_fight_key(history)

    history["snapshot_time"] = pd.to_datetime(history["snapshot_time"], errors="coerce", utc=True)
    history = history.dropna(subset=["snapshot_time"])
    history = _reorient_history(history, fighter_a)
    history = history.sort_values("snapshot_time").reset_index(drop=True)
    return history


def analyze_line_movement(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Analyze line movement for a specific fight.

    Returns dict with:
        - opening_prob_a: opening implied probability for fighter A
        - current_prob_a: current implied probability
        - movement: total probability shift (positive = moved toward A)
        - max_movement: largest single-snapshot move
        - is_sharp_move: True if line moved sharply (>5%)
        - direction: "toward_a", "toward_b", or "stable"
        - num_snapshots: number of data points
        - steam_move: True if all meaningful shifts moved in one direction
    """
    history = load_line_history(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    if history.empty:
        return compute_line_movement_analysis([])

    consensus = history.groupby("snapshot_time").agg(
        a_prob=("a_fair_prob", "mean"),
        b_prob=("b_fair_prob", "mean"),
    ).reset_index()

    result = compute_line_movement_analysis(
        consensus["a_prob"].tolist(),
        timestamps=consensus["snapshot_time"].tolist(),
    )

    if result.get("is_sharp_move"):
        logger.info(
            "SHARP LINE MOVE: %s vs %s | Opened %.1f%% -> Now %.1f%% (%+.1f%% %s)",
            fighter_a,
            fighter_b,
            result["opening_prob_a"] * 100,
            result["current_prob_a"] * 100,
            result["movement"] * 100,
            result["direction"],
        )
    if result.get("steam_move"):
        logger.warning(
            "STEAM MOVE DETECTED: %s vs %s | Consistent movement %s - likely sharp money",
            fighter_a,
            fighter_b,
            result["direction"],
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
    1. Extreme line movement (>15% shift from opening) - indicates sudden news
    2. One side near zero on Polymarket - fight is essentially off
    """
    from src.config import INJURY_MOVE_THRESHOLD, INJURY_PRICE_FLOOR

    result = {
        "suspected": False,
        "reason": "",
        "severity": "ok",
        "details": {},
    }

    analysis = analyze_line_movement(fighter_a, fighter_b)
    abs_move = abs(analysis.get("movement", 0))

    opening_a = analysis.get("opening_prob_a")
    current_a = analysis.get("current_prob_a")

    if abs_move >= INJURY_MOVE_THRESHOLD:
        direction = analysis.get("direction", "unknown")
        moved_away_from = fighter_b if direction == "toward_a" else fighter_a
        result["suspected"] = True
        result["severity"] = "block"
        result["reason"] = (
            f"The betting line has shifted {abs_move:.0%} away from {moved_away_from}. "
            f"A move this large usually means an injury, withdrawal, or fight cancellation. "
            f"Betting is blocked on this fight until the situation is confirmed."
        )
        if opening_a is not None and current_a is not None:
            result["reason"] += (
                f" Line moved from {opening_a:.0%}/{1-opening_a:.0%} "
                f"to {current_a:.0%}/{1-current_a:.0%}."
            )
        result["details"] = analysis
        logger.warning(
            "INJURY ALERT: %s vs %s - %.0f%% line shift detected",
            fighter_a,
            fighter_b,
            abs_move * 100,
        )
        return result

    if current_odds:
        a_prob = current_odds.get("a_prob", 0.5)
        b_prob = current_odds.get("b_prob", 0.5)

        if a_prob < INJURY_PRICE_FLOOR:
            result["suspected"] = True
            result["severity"] = "block"
            result["reason"] = (
                f"{fighter_a}'s market price has dropped to {a_prob:.0%}, "
                f"which is nearly zero. This typically means {fighter_a} has pulled out "
                f"of the fight or the bout has been cancelled. "
                f"Betting is blocked until this is resolved."
            )
            logger.warning("INJURY ALERT: %s", result["reason"])
            return result

        if b_prob < INJURY_PRICE_FLOOR:
            result["suspected"] = True
            result["severity"] = "block"
            result["reason"] = (
                f"{fighter_b}'s market price has dropped to {b_prob:.0%}, "
                f"which is nearly zero. This typically means {fighter_b} has pulled out "
                f"of the fight or the bout has been cancelled. "
                f"Betting is blocked until this is resolved."
            )
            logger.warning("INJURY ALERT: %s", result["reason"])
            return result

    if analysis.get("steam_move"):
        move = analysis.get("movement", 0)
        direction = analysis.get("direction", "unknown")

        if direction == "toward_a":
            favored = fighter_a
        elif direction == "toward_b":
            favored = fighter_b
        else:
            favored = "one side"

        result["suspected"] = True
        result["severity"] = "warning"
        result["reason"] = (
            f"The line is steadily moving toward {favored} ({abs(move):.1%} total shift). "
            f"Multiple consecutive moves in the same direction suggest sharp bettors "
            f"or insiders are backing {favored}."
        )
        if opening_a is not None and current_a is not None:
            result["reason"] += (
                f" Opened {opening_a:.0%}/{1-opening_a:.0%}, "
                f"now {current_a:.0%}/{1-current_a:.0%}."
            )
        result["details"] = analysis

    return result


def get_line_movement_features(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Get line movement features for the prediction model.

    These features capture where the "smart money" is going,
    which is one of the strongest signals in sports betting.
    """
    analysis = analyze_line_movement(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    return analysis_to_line_movement_features(analysis)


def line_history_health(snapshot_df: pd.DataFrame) -> dict:
    """Summarize line-history coverage for the fights in a snapshot."""
    if snapshot_df.empty:
        return {
            "tracked_fights": 0,
            "with_opening_line": 0,
            "with_two_snapshots": 0,
        }

    openers = _load_opening_lines()
    with_opening_line = 0
    with_two_snapshots = 0

    fights = snapshot_df.drop_duplicates(subset=["fight_key"])[
        ["fighter_a", "fighter_b", "event_id", "commence_time", "fight_key"]
    ]
    for _, fight in fights.iterrows():
        if fight["fight_key"] in openers:
            with_opening_line += 1
        history = load_line_history(
            fight["fighter_a"],
            fight["fighter_b"],
            event_id=fight.get("event_id", ""),
            commence_time=fight.get("commence_time", ""),
        )
        if not history.empty and history["snapshot_time"].nunique() >= 2:
            with_two_snapshots += 1

    return {
        "tracked_fights": len(fights),
        "with_opening_line": with_opening_line,
        "with_two_snapshots": with_two_snapshots,
    }


def run_line_tracking_pass() -> dict:
    """Run a complete line tracking pass - snapshot odds + Polymarket + analyze."""
    logger.info("Running line tracking pass...")

    odds_df = snapshot_odds()
    poly_df = snapshot_polymarket_prices()
    coverage = line_history_health(odds_df) if not odds_df.empty else {
        "tracked_fights": 0,
        "with_opening_line": 0,
        "with_two_snapshots": 0,
    }

    analyses = {}
    if not odds_df.empty:
        fights = odds_df.groupby(["fight_key", "fighter_a", "fighter_b", "event_id", "commence_time"]).first().reset_index()
        for _, fight in fights.iterrows():
            key = f"{fight['fighter_a']} vs {fight['fighter_b']}"
            analyses[key] = analyze_line_movement(
                fight["fighter_a"],
                fight["fighter_b"],
                event_id=fight.get("event_id", ""),
                commence_time=fight.get("commence_time", ""),
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "odds_records": len(odds_df),
        "polymarket_markets": len(poly_df),
        "fights_analyzed": len(analyses),
        "sharp_moves": sum(1 for analysis in analyses.values() if analysis.get("is_sharp_move")),
        "steam_moves": sum(1 for analysis in analyses.values() if analysis.get("steam_move")),
        "coverage": coverage,
        "analyses": analyses,
    }

    logger.info(
        "Line tracking: %s fights, %s sharp moves, %s steam moves, %s/%s with >=2 snapshots",
        len(analyses),
        summary["sharp_moves"],
        summary["steam_moves"],
        coverage["with_two_snapshots"],
        coverage["tracked_fights"],
    )

    return summary
