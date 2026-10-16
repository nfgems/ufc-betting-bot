"""Future-only policy for classifying newly imported official fight results.

Rows without ``RESULT_POLICY_VERSION`` / ``result_policy_version`` are legacy
data.  Callers must leave those rows on their historical target semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


RESULT_POLICY_VERSION = "ufcstats_incoming_v1"

RESULT_CLASS_NORMAL = "normal"
RESULT_CLASS_DQ = "dq"
RESULT_CLASS_DRAW = "draw"
RESULT_CLASS_NO_CONTEST = "no_contest"
RESULT_CLASS_UNKNOWN = "unknown"
RESULT_CLASS_CONTRADICTORY = "contradictory"

QUARANTINED_RESULT_CLASSES = frozenset(
    {RESULT_CLASS_UNKNOWN, RESULT_CLASS_CONTRADICTORY}
)

_KNOWN_STATUS_TOKENS = frozenset({"W", "L", "D", "NC"})
_DQ_RE = re.compile(r"\b(?:DQ|DISQUALIFICATION)\b", flags=re.IGNORECASE)


class ResultPolicyError(ValueError):
    """A marked incoming result cannot safely enter accepted data."""


@dataclass(frozen=True)
class OfficialResultClassification:
    """Canonical interpretation of the two official participant statuses."""

    result_class: str
    winner_side: str | None
    binary_target_eligible: bool
    quarantine_reason: str = ""

    @property
    def quarantined(self) -> bool:
        return self.result_class in QUARANTINED_RESULT_CLASSES


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _outcome_tokens(outcome: object) -> list[str]:
    text = _text(outcome).upper()
    return [part.strip() for part in text.split("/")] if text else []


def classify_official_result(
    outcome: object,
    method: object = "",
) -> OfficialResultClassification:
    """Classify an official two-status result without guessing.

    Disqualifications retain the official winner for recordkeeping, but are not
    eligible for the binary training target.  Draws and no contests have no
    winner.  Malformed or impossible status pairs are quarantined.
    """

    tokens = _outcome_tokens(outcome)
    if len(tokens) != 2 or any(token not in _KNOWN_STATUS_TOKENS for token in tokens):
        return OfficialResultClassification(
            RESULT_CLASS_UNKNOWN,
            None,
            False,
            "unknown_result_outcome",
        )

    if tokens in (["W", "L"], ["L", "W"]):
        winner_side = "a" if tokens[0] == "W" else "b"
        if _DQ_RE.search(_text(method)):
            return OfficialResultClassification(
                RESULT_CLASS_DQ,
                winner_side,
                False,
            )
        return OfficialResultClassification(
            RESULT_CLASS_NORMAL,
            winner_side,
            True,
        )
    if tokens == ["D", "D"]:
        return OfficialResultClassification(RESULT_CLASS_DRAW, None, False)
    if tokens == ["NC", "NC"]:
        return OfficialResultClassification(RESULT_CLASS_NO_CONTEST, None, False)
    return OfficialResultClassification(
        RESULT_CLASS_CONTRADICTORY,
        None,
        False,
        "contradictory_result_outcome",
    )


def has_result_policy_marker(value: object) -> bool:
    """Return whether a row opts into future-only result handling."""

    return bool(_text(value))


def parse_policy_boolean(value: object) -> bool | None:
    """Parse an optional persisted policy boolean without truthy-string bugs."""

    text = _text(value).casefold()
    if not text:
        return None
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ResultPolicyError(f"invalid binary-target eligibility value: {value!r}")


def validate_marked_result(
    outcome: object,
    method: object,
    *,
    policy_version: object,
    persisted_result_class: object = None,
    persisted_binary_target_eligible: object = None,
) -> OfficialResultClassification:
    """Validate and return the classification for a policy-marked row."""

    marker = _text(policy_version)
    if marker != RESULT_POLICY_VERSION:
        raise ResultPolicyError(f"unsupported result policy marker: {marker or '<blank>'}")

    classified = classify_official_result(outcome, method)
    if classified.quarantined:
        raise ResultPolicyError(classified.quarantine_reason)

    stored_class = _text(persisted_result_class).casefold()
    if not stored_class:
        raise ResultPolicyError("marked result is missing its classification")
    if stored_class != classified.result_class:
        raise ResultPolicyError(
            "result policy classification mismatch: "
            f"stored={stored_class!r} classified={classified.result_class!r}"
        )

    stored_eligibility = parse_policy_boolean(persisted_binary_target_eligible)
    if stored_eligibility is None:
        raise ResultPolicyError("marked result is missing binary-target eligibility")
    if (
        stored_eligibility != classified.binary_target_eligible
    ):
        raise ResultPolicyError(
            "result policy eligibility mismatch: "
            f"stored={stored_eligibility!r} "
            f"classified={classified.binary_target_eligible!r}"
        )
    return classified


def winner_for_side(
    winner_side: str | None,
    *,
    fighter_a: object,
    fighter_b: object,
) -> object | None:
    """Resolve the official winner name from a classified participant side."""

    if winner_side == "a":
        return fighter_a
    if winner_side == "b":
        return fighter_b
    return None
