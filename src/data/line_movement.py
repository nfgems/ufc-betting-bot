"""
Shared line movement constants and feature helpers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

LINE_MOVEMENT_FEATURE_COLS = [
    "line_movement",
    "line_abs_movement",
    "line_is_sharp",
    "line_steam_move",
    "line_direction_toward_a",
    "line_direction_toward_b",
]

LINE_MOVEMENT_SHARP_THRESHOLD = 0.05
LINE_MOVEMENT_STEAM_SHIFT_THRESHOLD = 0.005
LINE_MOVEMENT_DIRECTION_THRESHOLD = 0.02


def line_movement_nan_features() -> dict[str, float]:
    """Return the honest missing-value contract for line movement features."""
    return {col: np.nan for col in LINE_MOVEMENT_FEATURE_COLS}


def _empty_analysis(num_snapshots: int = 0) -> dict:
    """Return a NaN/unknown analysis payload for insufficient history."""
    return {
        "opening_prob_a": None,
        "current_prob_a": None,
        "opening_prob_b": None,
        "current_prob_b": None,
        "movement": np.nan,
        "abs_movement": np.nan,
        "max_single_shift": np.nan,
        "is_sharp_move": None,
        "steam_move": None,
        "direction": "unknown",
        "num_snapshots": int(num_snapshots),
        "hours_tracked": np.nan,
    }


def line_movement_direction(movement: float) -> str:
    """Map a movement value to the contract's direction buckets."""
    if pd.isna(movement):
        return "unknown"
    if movement > LINE_MOVEMENT_DIRECTION_THRESHOLD:
        return "toward_a"
    if movement < -LINE_MOVEMENT_DIRECTION_THRESHOLD:
        return "toward_b"
    return "stable"


def compute_line_movement_analysis(
    a_probs: Sequence[float],
    *,
    timestamps: Optional[Sequence] = None,
) -> dict:
    """
    Compute shared line movement analysis from a consensus A-side probability series.

    This is the canonical training/live implementation for sharp, steam, and
    direction semantics.
    """
    df = pd.DataFrame({"a_prob": list(a_probs)})
    if timestamps is not None:
        df["snapshot_time"] = list(timestamps)
        df = df.sort_values("snapshot_time")

    df["a_prob"] = pd.to_numeric(df["a_prob"], errors="coerce")
    df = df.dropna(subset=["a_prob"]).reset_index(drop=True)

    if len(df) < 2:
        return _empty_analysis(num_snapshots=len(df))

    opening_a = float(df["a_prob"].iloc[0])
    current_a = float(df["a_prob"].iloc[-1])
    movement = current_a - opening_a

    shifts = df["a_prob"].diff().dropna()
    max_shift = float(shifts.abs().max()) if not shifts.empty else np.nan
    steam_move = bool(
        len(shifts) >= 2
        and (
            (shifts > LINE_MOVEMENT_STEAM_SHIFT_THRESHOLD).all()
            or (shifts < -LINE_MOVEMENT_STEAM_SHIFT_THRESHOLD).all()
        )
    )

    hours_tracked = np.nan
    if "snapshot_time" in df.columns:
        times = pd.to_datetime(df["snapshot_time"], errors="coerce")
        if len(times) >= 2 and times.notna().all():
            hours_tracked = float((times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0)

    return {
        "opening_prob_a": opening_a,
        "current_prob_a": current_a,
        "opening_prob_b": float(1.0 - opening_a),
        "current_prob_b": float(1.0 - current_a),
        "movement": float(movement),
        "abs_movement": float(abs(movement)),
        "max_single_shift": max_shift,
        "is_sharp_move": bool(abs(movement) > LINE_MOVEMENT_SHARP_THRESHOLD),
        "steam_move": steam_move,
        "direction": line_movement_direction(movement),
        "num_snapshots": len(df),
        "hours_tracked": hours_tracked,
    }


def analysis_to_line_movement_features(analysis: dict) -> dict[str, float]:
    """Convert an analysis payload into the six model features."""
    movement = analysis.get("movement", np.nan)
    if pd.isna(movement):
        return line_movement_nan_features()

    is_sharp = analysis.get("is_sharp_move")
    steam_move = analysis.get("steam_move")
    direction = analysis.get("direction")

    return {
        "line_movement": float(movement),
        "line_abs_movement": float(analysis.get("abs_movement", abs(movement))),
        "line_is_sharp": np.nan if is_sharp is None else int(bool(is_sharp)),
        "line_steam_move": np.nan if steam_move is None else int(bool(steam_move)),
        "line_direction_toward_a": 1 if direction == "toward_a" else 0,
        "line_direction_toward_b": 1 if direction == "toward_b" else 0,
    }


def comparison_to_line_movement_features(
    opening_prob_a: Optional[float],
    current_prob_a: Optional[float],
    *,
    steam_move: Optional[bool] = None,
) -> dict[str, float]:
    """
    Build line movement features from a single opening/current comparison.

    Steam cannot be inferred from a one-comparison fallback, so callers should
    leave it as None to preserve NaN.
    """
    opening = pd.to_numeric(pd.Series([opening_prob_a]), errors="coerce").iloc[0]
    current = pd.to_numeric(pd.Series([current_prob_a]), errors="coerce").iloc[0]
    if pd.isna(opening) or pd.isna(current):
        return line_movement_nan_features()

    movement = float(current - opening)
    direction = line_movement_direction(movement)

    return {
        "line_movement": movement,
        "line_abs_movement": float(abs(movement)),
        "line_is_sharp": int(abs(movement) > LINE_MOVEMENT_SHARP_THRESHOLD),
        "line_steam_move": np.nan if steam_move is None else int(bool(steam_move)),
        "line_direction_toward_a": 1 if direction == "toward_a" else 0,
        "line_direction_toward_b": 1 if direction == "toward_b" else 0,
    }
