"""
Shared stance normalization for UFC feature engineering.
"""

from __future__ import annotations

import math


STANCE_ENCODING_MAP = {
    "orthodox": 0.0,
    "southpaw": 1.0,
    "switch": 2.0,
    "open stance": 3.0,
    "sideways": 4.0,
}


def normalize_stance_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split()).casefold()


def encode_stance(value: object) -> float:
    normalized = normalize_stance_label(value)
    if not normalized:
        return float("nan")
    return STANCE_ENCODING_MAP.get(normalized, math.nan)
