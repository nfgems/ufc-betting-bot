"""Utilities for side-A/side-B orientation handling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


SIDE_COLUMN_PAIRS: tuple[tuple[str, str], ...] = (
    ("fighter_a", "fighter_b"),
    ("prob_a", "prob_b"),
    ("no_odds_prob_a", "no_odds_prob_b"),
    ("opening_prob_a", "opening_prob_b"),
    ("closing_prob_a", "closing_prob_b"),
    ("line_direction_toward_a", "line_direction_toward_b"),
)

NEGATED_DIRECTIONAL_COLUMNS = frozenset(
    {
        "h2h_record_diff",
        "line_movement",
    }
)


def _nan_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index)


def has_ab_feature_pair(columns: Sequence[str]) -> bool:
    """Return True when a feature contract contains matched a_/b_ columns."""
    column_set = set(columns)
    return any(
        col.startswith("a_") and f"b_{col[2:]}" in column_set
        for col in column_set
    )


def has_directional_columns(columns: Sequence[str]) -> bool:
    """Return True when a feature contract changes meaning under A/B swap."""
    column_set = set(columns)
    if has_ab_feature_pair(columns):
        return True
    if any(col.startswith("diff_") for col in column_set):
        return True
    if column_set.intersection(NEGATED_DIRECTIONAL_COLUMNS):
        return True
    return any(left in column_set or right in column_set for left, right in SIDE_COLUMN_PAIRS)


def swap_directional_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Return a copy of *frame* with A/B-oriented columns swapped.

    Missing counterpart values stay NaN. This preserves observed data only: it
    never invents a neutral midpoint or other fallback value.
    """
    schema = set(columns or frame.columns)
    swapped = frame.copy()

    for col in list(schema):
        if not col.startswith("a_"):
            continue
        counterpart = f"b_{col[2:]}"
        if counterpart not in schema:
            continue
        left_values = frame[col].copy() if col in frame.columns else _nan_series(frame.index)
        right_values = (
            frame[counterpart].copy()
            if counterpart in frame.columns
            else _nan_series(frame.index)
        )
        swapped[col] = right_values
        swapped[counterpart] = left_values

    for left, right in SIDE_COLUMN_PAIRS:
        if left not in schema and right not in schema:
            continue
        left_values = frame[left].copy() if left in frame.columns else _nan_series(frame.index)
        right_values = frame[right].copy() if right in frame.columns else _nan_series(frame.index)
        swapped[left] = right_values
        swapped[right] = left_values

    for col in schema:
        if col.startswith("diff_") or col in NEGATED_DIRECTIONAL_COLUMNS:
            if col in frame.columns:
                swapped[col] = -pd.to_numeric(frame[col], errors="coerce")

    return swapped
