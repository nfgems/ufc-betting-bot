"""Shared event-context heuristics used by training and live inference."""

from __future__ import annotations

import math
import re
from numbers import Real


_LAS_VEGAS_TOKENS = (
    "las vegas, nevada, usa",
    "las vegas united states",
    "enterprise, nevada, united states",
    "enterprise, nevada, usa",
)
_APEX_TITLE_TOKENS = (
    "ufc fight night",
    "ufc on espn",
    "ufc on abc",
    "ufc vegas",
    "dana white's contender series",
    "dwcs",
)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def coerce_nullable_bool(value) -> bool | None:
    """Return an observed boolean, preserving malformed or missing values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        return None

    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def coerce_scheduled_rounds(value) -> int | None:
    """Return only an observed integer three- or five-round schedule."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    rounds = int(numeric)
    return rounds if rounds in {3, 5} else None


def infer_empty_arena(event_title=None, location=None) -> float:
    """
    Infer the empty-arena flag using the requested Apex-first heuristic.

    Rules:
    - explicit `Apex` / `Meta Apex` mentions -> empty
    - otherwise, Las Vegas UFC Fight Night style cards are treated as Apex
    - known non-Apex contexts resolve to 0
    - missing title/location resolves to NaN
    """
    title = _normalize_text(event_title)
    loc = _normalize_text(location)

    if "apex" in title or "apex" in loc:
        return 1.0

    # Without both fields a non-Apex inference is not observed. In particular,
    # a known city alone cannot prove the venue/card was not at the Apex.
    if not title or not loc:
        return float("nan")

    is_las_vegas = any(token in loc for token in _LAS_VEGAS_TOKENS)
    is_apex_style_card = any(token in title for token in _APEX_TITLE_TOKENS)
    if is_las_vegas and is_apex_style_card:
        return 1.0

    return 0.0
