"""Shared feature comparison rules for explicit historical replay outputs.

This module does no lookup, model work, source acquisition or output writing.
Retained replay outputs are evidence to compare, not evidence that this process
recomputed features. Fresh builder replay requires a separate explicit source
context; the older command's configured-source defaults are not used here.
"""

from __future__ import annotations

import math

import pandas as pd


TIME_AGED_FIELDS = {
    "days_since_last_fight", "layoff_log", "cage_rust",
    "age", "age_squared", "age_over_35", "age_under_25",
}
MISMATCH_VERDICTS = (
    "live_feature_missing", "live_nan_train_value", "live_value_train_nan", "value_diverge",
)
ODDS_PASSTHROUGH = [
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]


def expand_time_aged(columns: list[str]) -> set[str]:
    return {column for column in columns for root in TIME_AGED_FIELDS
            if column == root or column in {f"a_{root}", f"b_{root}", f"diff_{root}"}}


def classify(live_val, train_val, tol: float) -> str:
    """Preserve the replay command's numeric/NaN verdict meanings."""
    live = pd.to_numeric(pd.Series([live_val]), errors="coerce").iloc[0]
    train = pd.to_numeric(pd.Series([train_val]), errors="coerce").iloc[0]
    if pd.isna(live):
        return "match" if pd.isna(train) else "live_nan_train_value"
    if pd.isna(train):
        return "live_value_train_nan"
    return "match" if abs(float(live) - float(train)) <= tol else "value_diverge"


def numeric_value(value) -> float | None:
    """Missing stays missing; malformed or infinite values are never NaN matches."""
    if value is None or value is pd.NA or bool(pd.isna(value)):
        return None
    if isinstance(value, (str, bool)):
        raise ValueError("Feature contains a nonnumeric observed value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Feature contains a nonfinite observed value.")
    return number


def history_identity_key(row: pd.Series, side: str) -> str | None:
    """Retain the older replay helper's explicit ID/name compatibility."""
    from src.data.name_utils import fighter_identity_key

    raw_id = row.get(f"fighter_{side}_id")
    return fighter_identity_key(row.get(f"fighter_{side}"), raw_id if pd.notna(raw_id) else None)


def established_history_mask(frame: pd.DataFrame) -> pd.Series:
    """Both fighters must have history on a strictly earlier event date."""
    seen: set[str] = set()
    eligible = {}
    ordered = frame.sort_values("event_date", kind="mergesort")
    for _, same_date in ordered.groupby("event_date", sort=False):
        newly_seen: set[str] = set()
        for index, row in same_date.iterrows():
            identities = []
            for side in ("a", "b"):
                identities.append(history_identity_key(row, side))
            eligible[index] = all(identity is not None and identity in seen for identity in identities)
            newly_seen.update(identity for identity in identities if identity is not None)
        seen.update(newly_seen)
    return pd.Series(eligible, dtype=bool).reindex(frame.index, fill_value=False)
