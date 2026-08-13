"""Shared event-context heuristics used by training and live inference."""

from __future__ import annotations

import re


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

    if not title and not loc:
        return float("nan")
    if "apex" in title or "apex" in loc:
        return 1.0

    is_las_vegas = any(token in loc for token in _LAS_VEGAS_TOKENS)
    is_apex_style_card = any(token in title for token in _APEX_TITLE_TOKENS)
    if is_las_vegas and is_apex_style_card:
        return 1.0

    return 0.0
