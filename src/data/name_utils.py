"""
Shared fighter-name normalization and matching helpers.
"""

from __future__ import annotations

import re
import unicodedata

_SPECIAL_LETTER_TRANSLITERATION = str.maketrans(
    {
        "ß": "ss",
        "Æ": "AE",
        "æ": "ae",
        "Ø": "O",
        "ø": "o",
        "Đ": "D",
        "đ": "d",
        "Ð": "D",
        "ð": "d",
        "Ł": "L",
        "ł": "l",
        "Œ": "OE",
        "œ": "oe",
        "Þ": "TH",
        "þ": "th",
        "ı": "i",
        "Ħ": "H",
        "ħ": "h",
    }
)


def normalize_person_name(value: object) -> str:
    """Normalize a fighter name for durable cross-source matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_SPECIAL_LETTER_TRANSLITERATION)
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
    "zach": "zachary",
    "zack": "zachary",
}

# Exact spelling variants that also appear as standalone name tokens in market
# labels. Keep these separate from full-name aliases so surname-only rows such
# as "Buzukia wins by decision" canonicalize safely without fuzzy matching.
_FIGHTER_NAME_TOKEN_ALIASES: dict[str, str] = {
    "buzukia": "buzukja",
}

_FIGHTER_CANONICAL_ALIASES: dict[str, str] = {
    "asu almabaev": "asu almabayev",
    "assu almabaev": "asu almabayev",
    # The Odds API has emitted this apostrophe artifact for Ludovit Klein.
    # Punctuation normalization turns it into the otherwise-unmatchable
    # three-token key ``l udovit klein``.
    "l udovit klein": "ludovit klein",
    "ezra elliot": "ezra elliott",
    "seokhyeon ko": "seok hyun ko",
    "luis dias de assis": "luis felipe dias",
    "luis felipe dias de assis": "luis felipe dias",
    "nursultan ruziboev": "nursulton ruziboev",
    "patricio freire": "patricio pitbull",
    "patricio pitbull freire": "patricio pitbull",
    # Current-card sources use different legal/ring-name variants for these
    # fighters. Keep the aliases project-wide so official-card matching,
    # roster/history lookup, and UFCStats discovery agree on one identity.
    "carlos diego ferreira": "diego ferreira",
    "jose montanha da silva": "jose montanha",
    "william goff": "william ray goff",
    "yadier delvalle": "yadier del valle",
    # Bookmakers and several external sources omit his maternal surname.  Keep
    # this project-wide so UFCStats lookup, method-odds matching, rankings, and
    # cache identities all resolve the same fighter.
    "ian garry": "ian machado garry",
    # "Bobby" is expanded by _NICKNAME_MAP before this alias table is applied.
    "robert green": "king green",
}

_FIGHTER_DISPLAY_NAMES: dict[str, str] = {
    "asu almabayev": "Asu Almabayev",
    "king green": "King Green",
    "luis felipe dias": "Luis Felipe Dias",
    "nursulton ruziboev": "Nursulton Ruziboev",
    "ian machado garry": "Ian Machado Garry",
    "ezra elliott": "Ezra Elliott",
    "seok hyun ko": "Seok Hyun Ko",
    "dennis buzukja": "Dennis Buzukja",
    "ludovit klein": "Ludovit Klein",
    "diego ferreira": "Diego Ferreira",
    "jose montanha": "Jose Montanha",
    "william ray goff": "Billy Ray Goff",
    "yadier del valle": "Yadier del Valle",
}


def canonical_fighter_name_key(value: object) -> str:
    """Return the normalized canonical key for known cross-source aliases."""
    normalized = normalize_person_name(value)
    normalized = _NAME_SUFFIXES.sub("", normalized)
    tokens = normalized.split()
    if tokens:
        tokens[0] = _NICKNAME_MAP.get(tokens[0], tokens[0])
    tokens = [_FIGHTER_NAME_TOKEN_ALIASES.get(token, token) for token in tokens]
    normalized = " ".join(tokens)
    return _FIGHTER_CANONICAL_ALIASES.get(normalized, normalized)


def canonical_fighter_display_name(value: object) -> str:
    """Return the preferred display name when a source uses a known alias."""
    key = canonical_fighter_name_key(value)
    return _FIGHTER_DISPLAY_NAMES.get(key, str(value or "").strip())


def normalize_cross_source_name(value: object) -> str:
    """Aggressive normalization for matching the same fighter across sources.

    Strips suffixes (Jr, Sr, III …) and canonicalizes common first-name
    short forms so that "Joe Pyfer" and "Joseph Pyfer" produce the same key.
    """
    return canonical_fighter_name_key(value)


def person_name_tokens(value: object) -> list[str]:
    return [token for token in normalize_person_name(value).split() if token]


def _cross_source_name_tokens(value: object) -> list[str]:
    return [token for token in normalize_cross_source_name(value).split() if token]


def _tokens_match(query_tokens: list[str], candidate_tokens: list[str]) -> bool:
    return (
        bool(query_tokens)
        and bool(candidate_tokens)
        and (
            query_tokens == candidate_tokens
            or query_tokens == list(reversed(candidate_tokens))
        )
    )


def same_person_name(query: object, candidate: object) -> bool:
    """
    Match full fighter identities only.

    Token-order reversal is allowed for sources that flip Eastern/Western order,
    but extra/missing tokens are rejected to avoid same-surname collisions.
    """
    query_tokens = person_name_tokens(query)
    candidate_tokens = person_name_tokens(candidate)
    if _tokens_match(query_tokens, candidate_tokens):
        return True
    return _tokens_match(
        _cross_source_name_tokens(query),
        _cross_source_name_tokens(candidate),
    )


def name_appears_in_text(name: object, text: object) -> bool:
    """
    Check whether a full fighter name appears in a larger text snippet.

    This is stricter than substring matching because it compares normalized token
    spans instead of raw substrings.
    """
    text_tokens = person_name_tokens(text)
    if not text_tokens:
        return False

    window_sizes = {
        len(tokens)
        for tokens in (person_name_tokens(name), _cross_source_name_tokens(name))
        if tokens
    }
    for window_size in sorted(window_sizes):
        for start in range(len(text_tokens) - window_size + 1):
            if same_person_name(name, " ".join(text_tokens[start:start + window_size])):
                return True
    return False
