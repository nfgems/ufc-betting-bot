"""
Shared fighter-name normalization and matching helpers.
"""

from __future__ import annotations

import re
import unicodedata


# Small, source-reviewed identity set used only where name-keyed historical
# data would otherwise split or contaminate these exact fighters. This is not
# a suffix rule: only the spellings listed in ``tracked_names`` are joined.
REVIEWED_FIGHTER_IDENTITIES: dict[str, dict[str, object]] = {
    "f2f140ce7532e327": {
        "canonical_name": "Gabriel Santos",
        "tracked_names": ("Gabriel Santos",),
        "dob": "1996-11-28",
        "ufcstats_url": "http://ufcstats.com/fighter-details/f2f140ce7532e327",
        "sherdog_profile_id": "179211",
        "sherdog_url": "https://www.sherdog.com/fighter/Gabriel-Santos-179211",
        "tapology_profile_id": "159865",
        "tapology_url": (
            "https://www.tapology.com/fightcenter/fighters/"
            "159865-gabriel-santos-mosquitinho"
        ),
        "reviewed_history": {"professional": (10, 10, 0)},
    },
    "eee0ef3e2b14816b": {
        "canonical_name": "Kai Kamaka III",
        "tracked_names": ("Kai Kamaka", "Kai Kamaka III"),
        "dob": "1995-01-05",
        "ufcstats_url": "http://ufcstats.com/fighter-details/eee0ef3e2b14816b",
        "sherdog_profile_id": "117585",
        "sherdog_url": "https://www.sherdog.com/fighter/Kai-Kamaka-III-117585",
        "tapology_profile_id": "74838",
        "tapology_url": (
            "https://www.tapology.com/fightcenter/fighters/"
            "74838-kai-kamaka-iii-boy"
        ),
        "reviewed_history": {
            "professional": (9, 7, 2),
            "amateur": (3, 2, 1),
        },
    },
    "43a59ce3bb40449e": {
        "canonical_name": "Mizuki",
        "tracked_names": ("Mizuki", "Mizuki Inoue"),
        "dob": "1994-08-19",
        "ufcstats_url": "http://ufcstats.com/fighter-details/43a59ce3bb40449e",
        "sherdog_profile_id": "71390",
        "sherdog_url": "https://www.sherdog.com/fighter/Mizuki-Inoue-71390",
        "tapology_profile_id": "25717",
        "tapology_url": (
            "https://www.tapology.com/fightcenter/fighters/25717-mizuki-inoue"
        ),
        "reviewed_history": {"professional": (18, 13, 5)},
    },
}

_REVIEWED_EXACT_NAME_MAP = {
    tracked_name: str(identity["canonical_name"])
    for identity in REVIEWED_FIGHTER_IDENTITIES.values()
    for tracked_name in identity["tracked_names"]
}


def canonicalize_reviewed_fighter_name(value: object) -> object:
    """Canonicalize only exact spellings backed by a reviewed UFCStats ID."""
    if not isinstance(value, str):
        return value
    return _REVIEWED_EXACT_NAME_MAP.get(value, value)


def reviewed_fighter_identity_id(value: object) -> str | None:
    """Return the reviewed UFCStats ID without suffix-stripping the spelling."""
    if not isinstance(value, str):
        return None
    return _REVIEWED_NORMALIZED_ID_MAP.get(normalize_person_name(value))

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


_REVIEWED_NORMALIZED_ID_MAP = {
    normalize_person_name(tracked_name): ufcstats_id
    for ufcstats_id, identity in REVIEWED_FIGHTER_IDENTITIES.items()
    for tracked_name in identity["tracked_names"]
}


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


def _legacy_canonical_fighter_name_key(value: object) -> str:
    normalized = normalize_person_name(value)
    normalized = _NAME_SUFFIXES.sub("", normalized)
    tokens = normalized.split()
    if tokens:
        tokens[0] = _NICKNAME_MAP.get(tokens[0], tokens[0])
    tokens = [_FIGHTER_NAME_TOKEN_ALIASES.get(token, token) for token in tokens]
    normalized = " ".join(tokens)
    return _FIGHTER_CANONICAL_ALIASES.get(normalized, normalized)


_REVIEWED_CANONICAL_NAME_KEY_BY_ID = {
    ufcstats_id: normalize_person_name(identity["canonical_name"])
    for ufcstats_id, identity in REVIEWED_FIGHTER_IDENTITIES.items()
}
_REVIEWED_LEGACY_BASE_KEYS = frozenset(
    _legacy_canonical_fighter_name_key(tracked_name)
    for identity in REVIEWED_FIGHTER_IDENTITIES.values()
    for tracked_name in identity["tracked_names"]
)


def canonical_fighter_name_key(value: object) -> str:
    """Return the normalized canonical key for known cross-source aliases."""
    reviewed_id = reviewed_fighter_identity_id(value)
    if reviewed_id is not None:
        return _REVIEWED_CANONICAL_NAME_KEY_BY_ID[reviewed_id]

    legacy_key = _legacy_canonical_fighter_name_key(value)
    if legacy_key in _REVIEWED_LEGACY_BASE_KEYS:
        return normalize_person_name(value)
    return legacy_key


def canonical_fighter_display_name(value: object) -> str:
    """Return the preferred display name when a source uses a known alias."""
    key = canonical_fighter_name_key(value)
    return _FIGHTER_DISPLAY_NAMES.get(key, str(value or "").strip())


def normalize_cross_source_name(value: object) -> str:
    """Aggressive normalization for matching the same fighter across sources.

    Strips suffixes (Jr, Sr, III …) and canonicalizes common first-name
    short forms so that "Joe Pyfer" and "Joseph Pyfer" produce the same key.
    Source-reviewed identities retain suffixes where stripping would create a
    known collision.
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
    query_reviewed_id = reviewed_fighter_identity_id(query)
    candidate_reviewed_id = reviewed_fighter_identity_id(candidate)
    if query_reviewed_id is not None or candidate_reviewed_id is not None:
        return (
            query_reviewed_id is not None
            and query_reviewed_id == candidate_reviewed_id
        )

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
