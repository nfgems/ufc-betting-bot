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


_NAME_SUFFIXES = re.compile(
    r"\b(?:jr|sr|junior|senior|ii|iii|iv|v|2nd|3rd)\b", re.IGNORECASE,
)

# Common first-name short forms used by different odds/stats sources.
_NICKNAME_MAP: dict[str, str] = {
    "joe": "joseph",
    "joey": "joseph",
    "mike": "michael",
    "mikey": "michael",
    "alex": "alexander",
    "dan": "daniel",
    "danny": "daniel",
    "matt": "matthew",
    "rob": "robert",
    "bob": "robert",
    "bobby": "robert",
    "will": "william",
    "bill": "william",
    "billy": "william",
    "nick": "nicholas",
    "tony": "anthony",
    "chris": "christopher",
    "ed": "edward",
    "eddie": "edward",
    "ben": "benjamin",
    "benny": "benjamin",
    "greg": "gregory",
    "tom": "thomas",
    "tommy": "thomas",
    "tim": "timothy",
    "timmy": "timothy",
    "dave": "david",
    "jim": "james",
    "jimmy": "james",
    "pat": "patrick",
    "paddy": "patrick",
    "sam": "samuel",
    "sammy": "samuel",
    "jake": "jacob",
    "charlie": "charles",
    "chuck": "charles",
    "dick": "richard",
    "rick": "richard",
    "ricky": "richard",
    "steve": "steven",
    "stevie": "steven",
    "nate": "nathaniel",
    "vince": "vincent",
    "vinny": "vincent",
    "len": "leonard",
    "lenny": "leonard",
    "andy": "andrew",
    "drew": "andrew",
    "jack": "john",
    "johnny": "john",
    "jon": "john",
    "kenny": "kenneth",
    "larry": "lawrence",
    "marty": "martin",
    "ray": "raymond",
    "ronnie": "ronald",
    "ted": "theodore",
    "wes": "wesley",
}


def normalize_cross_source_name(value: object) -> str:
    """Aggressive normalization for matching the same fighter across sources.

    Strips suffixes (Jr, Sr, III …) and canonicalizes common first-name
    short forms so that "Joe Pyfer" and "Joseph Pyfer" produce the same key.
    """
    text = normalize_person_name(value)
    text = _NAME_SUFFIXES.sub("", text)
    tokens = text.split()
    if tokens:
        tokens[0] = _NICKNAME_MAP.get(tokens[0], tokens[0])
    return " ".join(tokens)


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
