"""
Shared fighter-name normalization and matching helpers.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_person_name(value: object) -> str:
    """Normalize a fighter name for durable cross-source matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def person_name_tokens(value: object) -> list[str]:
    return [token for token in normalize_person_name(value).split() if token]


def same_person_name(query: object, candidate: object) -> bool:
    """
    Match full fighter identities only.

    Token-order reversal is allowed for sources that flip Eastern/Western order,
    but extra/missing tokens are rejected to avoid same-surname collisions.
    """
    query_tokens = person_name_tokens(query)
    candidate_tokens = person_name_tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return False
    return (
        query_tokens == candidate_tokens
        or query_tokens == list(reversed(candidate_tokens))
    )


def name_appears_in_text(name: object, text: object) -> bool:
    """
    Check whether a full fighter name appears in a larger text snippet.

    This is stricter than substring matching because it compares normalized token
    spans instead of raw substrings.
    """
    name_tokens = person_name_tokens(name)
    text_tokens = person_name_tokens(text)
    if not name_tokens or not text_tokens:
        return False
    if len(name_tokens) < 2:
        return same_person_name(name, text)

    candidate_sequences = [name_tokens]
    if len(name_tokens) > 1:
        candidate_sequences.append(list(reversed(name_tokens)))

    for sequence in candidate_sequences:
        window = len(sequence)
        for start in range(len(text_tokens) - window + 1):
            if text_tokens[start:start + window] == sequence:
                return True
    return False
