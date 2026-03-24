"""Improve tennis odds matching by fixing name and tournament mismatches.

This script:
1. Loads matches_with_odds.csv and tennis_data_co_uk_odds.csv
2. Identifies unmatched rows that SHOULD have odds data
3. Uses fuzzy matching on names and tournaments to recover matches
4. Outputs a detailed report, saves an improved matches_with_odds.csv,
   and writes a modeling-safe export with quarantined bad rows
"""

import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ODDS_PATH = ROOT / "data" / "raw" / "tennis" / "tennis_data_co_uk_odds.csv"
MATCHES_BASE_PATH = ROOT / "data" / "processed" / "tennis" / "matches.csv"
MATCHES_BASE_MODELING_SAFE_PATH = ROOT / "data" / "processed" / "tennis" / "matches_modeling_safe.csv"
MATCHES_PATH = ROOT / "data" / "processed" / "tennis" / "matches_with_odds.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "tennis" / "matches_with_odds.csv"
MODELING_SAFE_OUTPUT_PATH = ROOT / "data" / "processed" / "tennis" / "matches_with_odds_modeling_safe.csv"
REPORT_PATH = ROOT / "data" / "processed" / "tennis" / "odds_matching_report.txt"
DIAGNOSTICS_PATH = ROOT / "data" / "processed" / "tennis" / "unmatched_diagnostics.csv"
CONFIRMED_BAD_ROWS_PATH = ROOT / "data" / "processed" / "tennis" / "known_bad_match_rows.csv"
SUSPECTED_BAD_ROWS_PATH = ROOT / "data" / "processed" / "tennis" / "suspected_bad_match_rows.csv"
BASE_CONFIRMED_BAD_ROWS_PATH = ROOT / "data" / "processed" / "tennis" / "known_bad_base_match_rows.csv"
BASE_SUSPECTED_BAD_ROWS_PATH = ROOT / "data" / "processed" / "tennis" / "suspected_bad_base_match_rows.csv"

SUSPECTED_UPSTREAM_BAD_MATCHES = [
    {
        "event_date": "2023-06-26",
        "tour": "wta",
        "tourney_name": "Bad Homburg",
        "round": "R32",
        "winner": "Leylah Fernandez",
        "loser_name": "Lena Rueffer",
        "reason": "No such Bad Homburg match was found; nearby history shows different Leylah Fernandez opponents.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "QF",
        "winner": "A. Cazaux",
        "loser_name": "K. Jacquet",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "R32",
        "winner": "C. Tabur",
        "loser_name": "A. Rinderknech",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "QF",
        "winner": "C. Tabur",
        "loser_name": "D. Altmaier",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "R32",
        "winner": "G. Mpetshi Perricard",
        "loser_name": "A. Blockx",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "SF",
        "winner": "T. Atmane",
        "loser_name": "A. Cazaux",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "QF",
        "winner": "T. Atmane",
        "loser_name": "C. Norrie",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "R32",
        "winner": "T. Atmane",
        "loser_name": "L. Sonego",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "QF",
        "winner": "V. Sachko",
        "loser_name": "A. Vukic",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "SF",
        "winner": "V. Sachko",
        "loser_name": "C. Tabur",
        "reason": "Metz 2025 bucket looks corrupted; the only nearby Sachko-Tabur match had different date/result context.",
    },
    {
        "event_date": "2025-11-02",
        "tour": "atp",
        "tourney_name": "Metz",
        "round": "F",
        "winner": "V. Sachko",
        "loser_name": "T. Atmane",
        "reason": "Metz 2025 bucket looks corrupted; no exact main-draw candidate was found.",
    },
    {
        "event_date": "2025-11-09",
        "tour": "atp",
        "tourney_name": "Nitto ATP Finals",
        "round": "RR",
        "winner": "B. Shelton",
        "loser_name": "C. Alcaraz",
        "reason": "No such round-robin event was found in the 2025 ATP Finals schedule.",
    },
    {
        "event_date": "2025-11-09",
        "tour": "atp",
        "tourney_name": "Nitto ATP Finals",
        "round": "RR",
        "winner": "F. Auger-Aliassime",
        "loser_name": "A. de Minaur",
        "reason": "No such round-robin event was found in the 2025 ATP Finals schedule.",
    },
    {
        "event_date": "2025-12-17",
        "tour": "atp",
        "tourney_name": "Next Gen ATP Finals",
        "round": "RR",
        "winner": "A. Blockx",
        "loser_name": "R. Jodar",
        "reason": "No such Next Gen event was found in the 2025 schedule.",
    },
    {
        "event_date": "2025-12-17",
        "tour": "atp",
        "tourney_name": "Next Gen ATP Finals",
        "round": "F",
        "winner": "J. Engel",
        "loser_name": "A. Blockx",
        "reason": "The real Engel-Blockx match was round robin, not a final.",
    },
]


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def make_match_row_key(event_date, tour, tourney_name, round_name, winner, loser_name):
    parsed_date = pd.to_datetime(event_date, errors="coerce")
    date_key = parsed_date.date().isoformat() if pd.notna(parsed_date) else ""
    return (
        date_key,
        normalize_text(tour),
        normalize_text(tourney_name),
        normalize_text(round_name),
        normalize_text(winner),
        normalize_text(loser_name),
    )


def parse_odds_name(odds_name):
    """Parse odds format name into (surname, initials_list).

    Examples:
        'Ramos-Vinolas A.' -> ('ramos vinolas', ['a'])
        'Varillas J.P.' -> ('varillas', ['j', 'p'])
        'Tseng C.H.' -> ('tseng', ['c', 'h'])
        'Cerundolo J.M.' -> ('cerundolo', ['j', 'm'])
    """
    norm = normalize_text(odds_name)
    tokens = norm.split()
    if not tokens:
        return "", []

    # Strip trailing single-char tokens as initials
    initials = []
    while len(tokens) > 1 and len(tokens[-1]) <= 2:
        initials.insert(0, tokens.pop()[0])

    surname = " ".join(tokens)
    return surname, initials


def extract_surname_from_odds_name(odds_name):
    surname, _ = parse_odds_name(odds_name)
    return surname


def extract_initial_from_odds_name(odds_name):
    _, initials = parse_odds_name(odds_name)
    return initials[0] if initials else ""


# ---- Tournament name mapping ----
# Maps (tour, normalized_our_name) -> normalized_odds_name or None (=excluded)

TOURNAMENT_ALIASES = {}

# ATP mappings
_atp_aliases = {
    "abn amro open": "abn amro world tennis tournament",
    "atp masters 1000 canada": "canadian open",
    "atp masters 1000 cincinnati": "western southern financial group masters",
    "atp masters 1000 paris": "bnp paribas masters",
    "atp masters 1000 rome": "internazionali bnl d italia",
    "acapulco": "abierto mexicano",
    "astana": "astana open",
    "atlanta": "atlanta open",
    "bci seguros chile open": "chile open",
    "banja luka": "srpska open",
    "barcelona": "barcelona open",
    "basel": "swiss indoors",
    "bastad": "nordea open",
    "beijing": "china open",
    "brisbane": "brisbane international",
    "buenos aires": "argentina open",
    "canada masters": "canadian open",
    "chengdu": "chengdu open",
    "cincinnati masters": "western southern financial group masters",
    "cordoba": "cordoba open",
    "dallas": "dallas open",
    "delray beach": "delray beach open",
    "doha": "qatar exxon mobil open",
    "dubai": "dubai tennis championships",
    "eastbourne": "eastbourne international",
    "estoril": "millennium estoril open",
    "firenze": "firenze open",
    "geneva": "geneva open",
    "gijon": "gijon open",
    "gstaad": "suisse open gstaad",
    "halle": "halle open",
    "hamburg": "hamburg open",
    "hangzhou": "hangzhou open",
    "hong kong": "hong kong tennis open",
    "houston": "u s men s clay court championships",
    "indian wells masters": "bnp paribas open",
    "kitzbuhel": "generali open",
    "korea open": "korea open",
    "los cabos": "los cabos open",
    "lyon": "lyon open",
    "madrid masters": "mutua madrid open",
    "mallorca": "mallorca championships",
    "marrakech": "grand prix hassan ii",
    "marseille": "open 13",
    "melbourne": "melbourne summer set",
    "miami masters": "miami open",
    "monte carlo masters": "monte carlo masters",
    "montpellier": "open sud de france",
    "munich": "bmw open",
    "naples": "napoli cup",
    "newport": "hall of fame championships",
    "paris masters": "bnp paribas masters",
    "pune": "maharashtra open",
    "queens club": "queen s club championships",
    "queen s club": "queen s club championships",
    "rio de janeiro": "rio open",
    "rio open presented by claro": "rio open",
    "rome masters": "internazionali bnl d italia",
    "rotterdam": "abn amro world tennis tournament",
    "s hertogenbosch": "rosmalen grass court championships",
    "san diego": "san diego open",
    "santiago": "chile open",
    "seoul": "korea open",
    "shanghai masters": "shanghai masters",
    "sofia": "sofia open",
    "st petersburg": "st petersburg open",
    "stockholm": "stockholm open",
    "stuttgart": "stuttgart open",
    "sydney": "sydney tennis classic",
    "tel aviv": "tel aviv open",
    "tokyo": "rakuten japan open tennis championships",
    "umag": "croatia open",
    "vienna": "vienna open",
    "washington": "citi open",
    "wimbledon": "wimbledon",
    "winston salem": "winston salem open at wake forest university",
    "zhuhai": "zhuhai championships",
    "metz": "open de moselle",
    "antwerp": "european open",
    "belgrade": "belgrade open",
    "belgrade 2": "belgrade open",
    "nitto atp finals": "masters cup",
    "auckland": "asb classic",
    "adelaide 1": "adelaide international 1",
    "adelaide 2": "adelaide international 2",
    "adelaide": "adelaide international",
    "melbourne 1": "melbourne summer set",
    "melbourne 2": "melbourne summer set",
    "ieb argentina open": "argentina open",
    "nexo dallas open": "dallas open",
    "open occitanie": "open sud de france",
    "roland garros": "french open",
    "us open": "us open",
    "australian open": "australian open",
    "atp masters 1000 indian wells": "bnp paribas open",
    "atp masters 1000 madrid": "mutua madrid open",
    "atp masters 1000 miami": "miami open",
    "atp masters 1000 monte carlo": "monte carlo masters",
    "atp masters 1000 shanghai": "shanghai masters",
    "tour finals": "masters cup",
    "florence": "firenze open",
    "london queen s club": "queen s club championships",
    "london queens club": "queen s club championships",
    "dubai duty free tennis championships": "dubai tennis championships",
    "qatar exxonmobil open": "qatar exxon mobil open",
    "belgrade 2": "belgrade open",
    "hamburg open": "hamburg open",
    "stockholm open": "stockholm open",
    "stuttgart open": "stuttgart open",
}

_wta_aliases = {
    "abu dhabi": "abu dhabi wta women s tennis open",
    "adelaide": "adelaide international",
    "adelaide 1": "adelaide international 1",
    "adelaide 2": "adelaide international 2",
    "auckland": "asb classic",
    "austin": "atx open",
    "bad homburg": "bad homburg open",
    "beijing": "china open",
    "birmingham": "rothesay classic",
    "bogota": "copa colsanitas",
    "bogota open": "copa colsanitas",
    "brisbane": "brisbane international",
    "budapest": "budapest open",
    "charleston": "charleston open",
    "chennai": "chennai open",
    "cincinnati": "western southern financial group masters",
    "cleveland": "tennis in the land",
    "cluj napoca": "transylvania open",
    "cluj-napoca": "transylvania open",
    "doha": "qatar open",
    "dubai": "dubai duty free tennis championships",
    "eastbourne": "eastbourne international",
    "granby": "championnats de granby",
    "guadalajara": "guadalajara open",
    "guadalajara 2": "guadalajara open",
    "guadalajara 500": "guadalajara open",
    "guangzhou": "guangzhou open",
    "hamburg": "hamburg open",
    "hobart": "hobart international",
    "hong kong": "hong kong tennis open",
    "hua hin": "thailand open",
    "indian wells": "bnp paribas open",
    "indian wells masters": "bnp paribas open",
    "istanbul": "istanbul cup",
    "lyon": "lyon open",
    "madrid": "mutua madrid open",
    "melbourne 1": "melbourne summer set 1",
    "melbourne 2": "melbourne summer set 2",
    "merida": "merida open",
    "m rida": "merida open",
    "miami": "miami open",
    "miami masters": "miami open",
    "monastir": "jasmin open",
    "nanchang": "jiangxi open",
    "nanchang open": "jiangxi open",
    "monterrey": "monterrey open",
    "montreal": "canadian open",
    "ningbo": "ningbo open",
    "nottingham": "nottingham open",
    "osaka": "japan open",  # Toray Pan Pacific 2022, Japan Open 2023+
    "palermo": "internazionali femminili di palermo",
    "prague": "prague open",
    "rabat": "morocco open",
    "rome": "internazionali bnl d italia",
    "san diego": "san diego open",
    "san jose": "mubadala silicon valley classic",
    "strasbourg": "internationaux de strasbourg",
    "washington": "citi open",
    "seoul": "korea open",
    "singapore": "singapore open",
    "st petersburg": "st petersburg ladies trophy",
    "stuttgart": "porsche tennis grand prix",
    "s hertogenbosch": "rosmalen grass court championships",
    "tallinn": "tallinn open",
    "toronto": "canadian open",
    "washington": "citi open",
    "wuhan": "wuhan open",
    "zhengzhou": "zhengzhou open",
    "roland garros": "french open",
    "us open": "us open",
    "australian open": "australian open",
    "wimbledon": "wimbledon",
    "tokyo": "toray pan pacific open tennis tournament",
    "berlin": "german open",  # German Open in 2023+; Bett1Open in 2022
    "linz": "ladies linz open",
    "lausanne": "ladies open lausanne",
    "strasbourg": "internationaux de strasbourg",
    "parma": "parma ladies open",
    "portoroz": "slovenia open",
    "warsaw": "poland open",
    "ostrava": "ostrava open",
    "iasi": "iasi open",
    "rouen": "open de rouen",
    "st petersburg": "ladies trophy",
    "washington dc": "citi open",
    "queens": "queen s club championships",
    "fort worth finals": "wta finals",
    "cancun finals": "wta finals",
    "riyadh finals": "wta finals",
    "jiujiang": "jiangxi open",
    "sao paulo": "sp open",
    "hua hin 2": "thailand open 2",
    "dubai duty free tennis championships": "dubai duty free tennis championships",
}

for name, target in _atp_aliases.items():
    TOURNAMENT_ALIASES[("atp", name)] = target
for name, target in _wta_aliases.items():
    TOURNAMENT_ALIASES[("wta", name)] = target

# Year-specific tournament alias overrides: (tour, name, year) -> odds_name
TOURNAMENT_YEAR_ALIASES = {
    ("wta", "berlin", 2022): "bett1open",
}


def build_tournament_lookup(odds_df):
    """Build {(tour, year): {normalized_name: original_name}}"""
    lookup = {}
    for _, row in odds_df.drop_duplicates(subset=["tour", "Tournament", "source_year"]).iterrows():
        key = (row["tour"], int(row["source_year"]))
        if key not in lookup:
            lookup[key] = {}
        lookup[key][normalize_text(row["Tournament"])] = row["Tournament"]
    return lookup


def find_odds_tournament(our_name, tour, year, odds_lookup):
    """Try to match our tournament name to an odds tournament name."""
    norm = normalize_text(our_name)
    available = odds_lookup.get((tour, year), {})

    # 1. Exact normalized match
    if norm in available:
        return available[norm]

    # 2a. Year-specific alias override
    year_alias = TOURNAMENT_YEAR_ALIASES.get((tour, norm, year))
    if year_alias is not None:
        if year_alias in available:
            return available[year_alias]

    # 2. Known alias
    alias_target = TOURNAMENT_ALIASES.get((tour, norm))
    if alias_target is not None:
        if alias_target in available:
            return available[alias_target]
        # Try partial match of alias in available
        for odds_norm, odds_orig in available.items():
            if alias_target in odds_norm or odds_norm in alias_target:
                return odds_orig

    # 3. Try alias with partial name match (e.g. "Eastbourne" alias target
    #    "eastbourne international" might not be there but "eastbourne open" is)
    if alias_target is not None:
        alias_main_tokens = set(alias_target.split()) - {
            "open", "masters", "international", "cup", "grand", "prix",
            "championships", "classic", "trophy",
        }
        if alias_main_tokens:
            for odds_norm, odds_orig in available.items():
                odds_tokens = set(odds_norm.split())
                if alias_main_tokens.issubset(odds_tokens):
                    return odds_orig

    # 4. Substring matching
    for odds_norm, odds_orig in available.items():
        if len(norm) >= 4 and (norm in odds_norm or odds_norm in norm):
            return odds_orig

    # 5. Token overlap - all tokens of our short name in odds name
    our_tokens = set(norm.split())
    # Remove common filler words
    our_tokens -= {"open", "masters", "international", "cup", "grand", "prix", "championships"}
    if our_tokens:
        for odds_norm, odds_orig in available.items():
            odds_tokens = set(odds_norm.split())
            if our_tokens.issubset(odds_tokens) and len(our_tokens) >= 1:
                return odds_orig

    return None


STRICT_TOURNAMENT_SOURCE_LABELS = {
    "tennis_data",
    "masterscup_local",
    "wayback_tennis",
}


def tournament_norm_candidates(name, tour, year=None):
    norm = normalize_text(name)
    if not norm:
        return set()

    candidates = {norm}
    alias_target = TOURNAMENT_ALIASES.get((tour, norm))
    if alias_target:
        candidates.add(alias_target)

    if year is not None:
        year_alias = TOURNAMENT_YEAR_ALIASES.get((tour, norm, year))
        if year_alias:
            candidates.add(year_alias)

    return {candidate for candidate in candidates if candidate}


def source_candidate_tournament_compatible(row, source_label, candidate):
    if source_label not in STRICT_TOURNAMENT_SOURCE_LABELS:
        return True

    event_date = row.get("event_date")
    year = int(event_date.year) if pd.notna(event_date) else None
    tour = row.get("tour", "")
    row_candidates = tournament_norm_candidates(row.get("tourney_name", ""), tour, year)
    source_candidates = tournament_norm_candidates(candidate.get("tournament", ""), tour, year)

    if not row_candidates or not source_candidates:
        return True

    for left in row_candidates:
        for right in source_candidates:
            if left == right:
                return True
            if len(left) >= 4 and len(right) >= 4 and (left in right or right in left):
                return True

    return False


ROUND_ALIASES = {
    "rr": "RR",
    "round robin": "RR",
    "qf": "QF",
    "quarterfinal": "QF",
    "quarter finals": "QF",
    "quarterfinals": "QF",
    "quarter final": "QF",
    "quarter finals": "QF",
    "semi final": "SF",
    "semi finals": "SF",
    "semifinal": "SF",
    "semifinals": "SF",
    "sf": "SF",
    "f": "F",
    "final": "F",
    "the final": "F",
    "1 64 finals": "R128",
    "1 32 finals": "R64",
    "1 16 finals": "R32",
    "1 8 finals": "R16",
    "1 4 finals": "QF",
    "1 2 finals": "SF",
    "round of 128": "R128",
    "round of 64": "R64",
    "round of 32": "R32",
    "round of 16": "R16",
}


def normalize_round_name(value):
    norm = normalize_text(value)
    if not norm:
        return ""

    direct = ROUND_ALIASES.get(norm)
    if direct:
        return direct

    match = re.fullmatch(r"r(\d+)", norm)
    if match:
        return f"R{match.group(1)}"

    return ""


def source_candidate_round_compatible(row, candidate):
    row_round = normalize_round_name(row.get("round", ""))
    candidate_round = normalize_round_name(candidate.get("round", candidate.get("Round", "")))

    if not row_round or not candidate_round:
        return True
    return row_round == candidate_round


# ---- Player name matching ----

# Build alias map dynamically from odds data
def build_player_alias_map(odds_df):
    """Build a map of odds player surnames+initials for fast lookup."""
    player_map = {}  # (tour, surname, initial) -> set of full odds names
    for _, row in odds_df.iterrows():
        tour = row["tour"]
        for name_col in ["Winner", "Loser"]:
            name = str(row[name_col])
            surname = extract_surname_from_odds_name(name)
            initial = extract_initial_from_odds_name(name)
            if surname:
                key = (tour, surname, initial)
                if key not in player_map:
                    player_map[key] = set()
                player_map[key].add(name)
    return player_map


# Hard-coded aliases for known problem cases
PLAYER_NAME_ALIASES = {
    # Our full name -> odds surname (normalized)
    "alison riske amritraj": "riske",
    "alison riske": "riske",
    "albert ramos": "ramos vinolas",
    "albert ramos vinolas": "ramos vinolas",
    "tomas barrios vera": "barrios",
    "bu yunchaokete": "bu",
    "felipe meligeni alves": "meligeni alves",
    "felipe meligeni rodrigues alves": "meligeni alves",
    "alejandro davidovich fokina": "davidovich fokina",
    "roberto bautista": "bautista agut",
    "roberto bautista agut": "bautista agut",
    "alex de minaur": "de minaur",
    "christopher oconnell": "o connell",
    "christopher o connell": "o connell",
    "marc andrea huesler": "huesler",
    "bernabe zapata miralles": "zapata miralles",
    "pablo carreno busta": "carreno busta",
    "pablo carreno": "carreno busta",
    "camilo ugo carabelli": "ugo carabelli",
    "tomas martin etcheverry": "etcheverry",
    "juan manuel cerundolo": "cerundolo j m",
    "francisco cerundolo": "cerundolo f",
    "beatriz haddad maia": "haddad maia",
    "jessica bouzas maneiro": "bouzas maneiro",
    "nuria parrizas diaz": "parrizas diaz",
    "sara sorribes tormo": "sorribes tormo",
    "paula badosa": "badosa",
    "paula badosa gibert": "badosa",
    "irina camelia begu": "begu",
    "irina begu": "begu",
    "jaqueline cristian": "cristian",
    "jacqueline cristian": "cristian",
    "tatjana maria": "maria",
    "mirra andreeva": "andreeva m",
    "erika andreeva": "andreeva e",
    "linda fruhvirtova": "fruhvirtova l",
    "brenda fruhvirtova": "fruhvirtova b",
    "karolina pliskova": "pliskova ka",
    "kristyna pliskova": "pliskova kr",
    "xiyu wang": "wang xiy",
    "xinyu wang": "wang xin",
    "xin yu wang": "wang xin",
    "qiang wang": "wang q",
    "storm hunter": "hunter",
    "storm sanders": "sanders",
    "camila osorio": "osorio",
    "camila osorio serrano": "osorio",
    "maria camila osorio serrano": "osorio",
    "chun hsin tseng": "tseng",
    "juan pablo varillas": "varillas",
    "panna udvardy": "udvardy",
    "viktoria hruncakova": "hruncakova",
    "cristian garin": "garin",
    "pedro martinez": "martinez",
    "pedro martinez portero": "martinez",
    "jaume munar": "munar",
    "jaume munar clar": "munar",
    "luca van assche": "van assche",
    "leylah fernandez": "fernandez",
    "leylah annie fernandez": "fernandez",
    "bianca vanessa andreescu": "andreescu",
    "bianca andreescu": "andreescu",
    "anna karolina schmiedlova": "schmiedlova",
    "despina papamichail": "papamichail",
    "valentini grammatikopoulou": "grammatikopoulou",
    "joanne zuger": "zuger",
    "lena rueffer": "rueffer",
    "margarita gasparyan": "gasparyan",
    "kateryna bondarenko": "bondarenko",
    "laura samson": "samson",
    "tara wurth": "wurth",
    "han shi": "shi",
    "antonia ruzic": "ruzic",
    "mariana isabel higuita barraza": "higuita barraza",
    "aliona bolsova": "bolsova",
    "irene burillo": "burillo",
    "viktoria morvayova": "morvayova",
    "hong yi cody wong": "wong",
    "felipe meligeni rodrigues alves": "meligeni rodrigues",
    "elsa jacquemot": "jacquemot",
    "darja vidmanova": "vidmanova",
    "astra sharma": "sharma",
    "ariana geerlings": "geerlings",
    "simona waltert": "waltert",
}


# Players who appear under multiple surnames in odds data (e.g., married name changes)
PLAYER_ALTERNATE_SURNAMES = {
    "storm hunter": ["hunter", "sanders"],
    "storm sanders": ["sanders", "hunter"],
    "camila osorio": ["osorio"],
    "camila osorio serrano": ["osorio"],
    "alison riske amritraj": ["riske", "riske amritraj"],
    "alison riske": ["riske"],
    "tomas barrios vera": ["barrios"],
    "viktoria hruncakova": ["hruncakova", "kuzmova"],
    "aliona bolsova": ["bolsova", "bolsova zadoinov"],
    "tara wurth": ["wurth", "wuerth"],
    "ariana geerlings": ["geerlings", "geerlings martinez"],
    "paula badosa": ["badosa"],
    "paula badosa gibert": ["badosa"],
    "felipe meligeni alves": ["meligeni alves", "meligeni rodrigues"],
    "felipe meligeni rodrigues alves": ["meligeni alves", "meligeni rodrigues"],
    "hong yi cody wong": ["wong", "wong hong yi", "wong hong"],
}


def _parse_abbreviated_name(name_str):
    """Detect and parse 2025-style abbreviated names like 'J. Faria' or 'T. Barrios Vera'.

    Returns (surname, initial) if abbreviated, else (None, None).
    """
    # Match pattern: single letter followed by period, then surname
    # e.g. "J. Faria", "T. Barrios Vera", "D. Dedura"
    m = re.match(r'^([A-Za-z])\.\s+(.+)$', str(name_str).strip())
    if m:
        initial = m.group(1).lower()
        surname = normalize_text(m.group(2))
        return surname, initial
    return None, None


def names_match_flexible(our_name, odds_name):
    """Check if our full name matches an odds 'Surname I.' name."""
    our_norm = normalize_text(our_name)
    odds_norm = normalize_text(odds_name)

    if not our_norm or not odds_norm:
        return False
    if our_norm == odds_norm:
        return True

    our_tokens = our_norm.split()
    if not our_tokens:
        return False

    odds_surname, odds_initials = parse_odds_name(odds_name)
    if not odds_surname:
        return False

    odds_initial = odds_initials[0] if odds_initials else ""

    # Handle 2025-style abbreviated names: "J. Faria", "T. Barrios Vera"
    abbrev_surname, abbrev_initial = _parse_abbreviated_name(our_name)
    if abbrev_surname is not None:
        initial_ok = (not odds_initial or odds_initial == abbrev_initial)
        if abbrev_surname == odds_surname and initial_ok:
            return True
        # For multi-word abbreviated surnames like "Barrios Vera",
        # try matching just the first token (e.g. "barrios" == "barrios")
        abbrev_tokens = abbrev_surname.split()
        if len(abbrev_tokens) > 1 and abbrev_tokens[0] == odds_surname and initial_ok:
            return True
        # Also check aliases for the abbreviated form
        alias_surname = PLAYER_NAME_ALIASES.get(our_norm)
        if alias_surname and normalize_text(alias_surname) == odds_surname:
            return True
        # Check if odds surname starts with abbreviated surname or vice versa
        # e.g. "dedura" vs "dedura palomero"
        if abbrev_surname in odds_surname and initial_ok:
            return True

    our_first_initial = our_tokens[0][0]
    # Collect all initials from our name tokens for lenient matching
    our_all_initials = {t[0] for t in our_tokens}

    def _initial_ok(odds_init, our_tokens_used_for_surname_start=0):
        """Check if odds initial is compatible with our name.

        Odds data sometimes uses middle name initial rather than first name,
        so we accept if the initial matches ANY token's first letter.
        """
        if not odds_init:
            return True
        # Check first name initial
        if odds_init == our_first_initial:
            return True
        # Check any name token that's NOT part of the surname
        for j in range(our_tokens_used_for_surname_start):
            if our_tokens[j][0] == odds_init:
                return True
        # Lenient: check any token
        if odds_init in our_all_initials:
            return True
        return False

    # Check known aliases first — skip initial check since alias is authoritative
    alias_surname = PLAYER_NAME_ALIASES.get(our_norm)
    if alias_surname:
        alias_norm = normalize_text(alias_surname)
        if alias_norm == odds_surname:
            return True
        if alias_norm == odds_norm:
            return True

    # Check alternate surnames (for married name changes, etc.)
    alt_surnames = PLAYER_ALTERNATE_SURNAMES.get(our_norm, [])
    for alt in alt_surnames:
        if normalize_text(alt) == odds_surname and _initial_ok(odds_initial):
            return True

    odds_surname_tokens = odds_surname.split()

    # Handle shortened surnames against compound source surnames.
    # Example: "Irene Burillo" should match "Burillo Escorihuela I."
    our_last = our_tokens[-1]
    if len(odds_surname_tokens) > 1 and our_last in {odds_surname_tokens[0], odds_surname_tokens[-1]}:
        if _initial_ok(odds_initial, len(our_tokens) - 1):
            return True

    # Strategy 1: Last token of our name == odds surname, initial compatible
    if our_last == odds_surname and _initial_ok(odds_initial, len(our_tokens) - 1):
        return True

    # Strategy 2: Multi-word surname in odds - match against tail of our name
    if len(odds_surname_tokens) > 1:
        for start in range(len(our_tokens)):
            candidate = " ".join(our_tokens[start:])
            if candidate == odds_surname and _initial_ok(odds_initial, start):
                return True

    # Strategy 3: Our name without first token matches odds surname
    if len(our_tokens) >= 2:
        our_surname_part = " ".join(our_tokens[1:])
        if our_surname_part == odds_surname and _initial_ok(odds_initial, 1):
            return True

    # Strategy 4: Handle "Firstname Middlename Surname" where odds has "Surname F."
    if len(our_tokens) >= 2:
        for i in range(1, len(our_tokens)):
            if our_tokens[i] == odds_surname and _initial_ok(odds_initial, i):
                return True

    # Strategy 5: For names like "Felix Auger Aliassime", try matching compound
    # surname from consecutive tokens against odds surname
    if len(our_tokens) >= 3:
        for start in range(1, len(our_tokens) - 1):
            for end in range(start + 2, len(our_tokens) + 1):
                candidate = " ".join(our_tokens[start:end])
                if candidate == odds_surname and _initial_ok(odds_initial, start):
                    return True

    return False


def try_match_row(match_row, odds_df_filtered, event_date, max_days):
    """Try to find odds data for an unmatched match row within filtered odds.

    Returns (odds_row_series, match_method) or (None, reason_for_failure)
    """
    winner = match_row["winner"]
    loser = match_row["loser_name"]

    if odds_df_filtered.empty:
        return None, "no_odds_for_tournament"

    # Filter by date proximity
    date_filtered = odds_df_filtered[
        (odds_df_filtered["Date"] - event_date).abs().dt.days <= max_days
    ]

    if date_filtered.empty:
        return None, "no_odds_in_date_range"

    # Pass 1: strict name matching
    best_match = None
    best_date_diff = 999

    for _, odds_row in date_filtered.iterrows():
        if not source_candidate_round_compatible(match_row, odds_row):
            continue
        w_match = names_match_flexible(winner, odds_row["Winner"])
        l_match = names_match_flexible(loser, odds_row["Loser"])
        if w_match and l_match:
            date_diff = abs((odds_row["Date"] - event_date).days)
            if date_diff < best_date_diff:
                best_date_diff = date_diff
                best_match = odds_row
                return best_match, "winner_loser_match"

        w_match_rev = names_match_flexible(winner, odds_row["Loser"])
        l_match_rev = names_match_flexible(loser, odds_row["Winner"])
        if w_match_rev and l_match_rev:
            date_diff = abs((odds_row["Date"] - event_date).days)
            if date_diff < best_date_diff:
                best_date_diff = date_diff
                best_match = odds_row
                return best_match, "winner_loser_swapped"

    if best_match is not None:
        return best_match, "matched"

    # Pass 2: surname-only matching with disambiguation
    # For cases where initial doesn't match (e.g., "Camila Osorio" vs "Osorio M.")
    # Only accept if there's exactly ONE candidate with that surname pair
    winner_surname = _extract_match_surname(winner)
    loser_surname = _extract_match_surname(loser)

    surname_candidates_normal = []
    surname_candidates_swapped = []

    for _, odds_row in date_filtered.iterrows():
        if not source_candidate_round_compatible(match_row, odds_row):
            continue
        w_odds_surname = extract_surname_from_odds_name(odds_row["Winner"])
        l_odds_surname = extract_surname_from_odds_name(odds_row["Loser"])

        if winner_surname == w_odds_surname and loser_surname == l_odds_surname:
            surname_candidates_normal.append(odds_row)
        elif winner_surname == l_odds_surname and loser_surname == w_odds_surname:
            surname_candidates_swapped.append(odds_row)

    if len(surname_candidates_normal) == 1:
        return surname_candidates_normal[0], "surname_only_match"
    if len(surname_candidates_swapped) == 1:
        return surname_candidates_swapped[0], "surname_only_swapped"

    return None, "name_mismatch"


def _extract_match_surname(full_name):
    """Extract the primary surname from our full name format for loose matching."""
    norm = normalize_text(full_name)
    tokens = norm.split()
    if not tokens:
        return ""

    # Handle 2025 abbreviated names: "J. Faria" -> surname "faria"
    # "T. Barrios Vera" -> use first token "barrios" (odds uses "Barrios M.")
    abbrev_surname, _ = _parse_abbreviated_name(full_name)
    if abbrev_surname is not None:
        abbrev_tokens = abbrev_surname.split()
        # Return first token for multi-word, or full for single-word
        return abbrev_tokens[0]

    # Check aliases first
    alias = PLAYER_NAME_ALIASES.get(norm)
    if alias:
        alias_norm = normalize_text(alias)
        # Return just the first token of the alias if multi-word
        return alias_norm.split()[0] if alias_norm else tokens[-1]

    # Check alternate surnames
    alts = PLAYER_ALTERNATE_SURNAMES.get(norm, [])
    if alts:
        return normalize_text(alts[0])

    # Default: last token
    return tokens[-1]


def assign_odds_to_row(match_row, odds_row, method):
    """Determine which odds go to player_a and player_b."""
    winner = match_row["winner"]
    player_a = match_row["player_a"]

    # Determine if odds Winner = our winner
    if method == "winner_loser_match":
        w_is_our_winner = True
    elif method == "winner_loser_swapped":
        w_is_our_winner = False
    else:
        w_is_our_winner = names_match_flexible(winner, odds_row["Winner"])

    # Map: player_a might be winner or loser
    a_is_winner = (player_a == winner)

    # Determine if player_a maps to odds Winner
    if a_is_winner == w_is_our_winner:
        # player_a = odds Winner
        return {
            "b365_a": _safe_float(odds_row.get("B365W")),
            "b365_b": _safe_float(odds_row.get("B365L")),
            "ps_a": _safe_float(odds_row.get("PSW")),
            "ps_b": _safe_float(odds_row.get("PSL")),
            "max_a": _safe_float(odds_row.get("MaxW")),
            "max_b": _safe_float(odds_row.get("MaxL")),
            "avg_a": _safe_float(odds_row.get("AvgW")),
            "avg_b": _safe_float(odds_row.get("AvgL")),
        }
    else:
        # player_a = odds Loser
        return {
            "b365_a": _safe_float(odds_row.get("B365L")),
            "b365_b": _safe_float(odds_row.get("B365W")),
            "ps_a": _safe_float(odds_row.get("PSL")),
            "ps_b": _safe_float(odds_row.get("PSW")),
            "max_a": _safe_float(odds_row.get("MaxL")),
            "max_b": _safe_float(odds_row.get("MaxW")),
            "avg_a": _safe_float(odds_row.get("AvgL")),
            "avg_b": _safe_float(odds_row.get("AvgW")),
        }


def _safe_float(val):
    """Convert to float, returning NaN for unparseable values."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def has_any_odds_mask(df):
    return (
        df["b365_a"].notna() | df["ps_a"].notna() |
        df["max_a"].notna() | df["avg_a"].notna()
    )


def _coalesce_name(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _names_equivalent(name_a, name_b):
    norm_a = normalize_text(name_a)
    norm_b = normalize_text(name_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    return names_match_flexible(name_a, name_b) or names_match_flexible(name_b, name_a)


def build_row_identity_variants(row):
    winner = _coalesce_name(row.get("winner"))
    loser = _coalesce_name(row.get("loser_name"))
    player_a = _coalesce_name(row.get("player_a"))
    player_b = _coalesce_name(row.get("player_b"))

    variants = []
    seen = set()

    def add_variant(w_name, l_name, tag):
        w_norm = normalize_text(w_name)
        l_norm = normalize_text(l_name)
        if not w_norm or not l_norm or w_norm == l_norm:
            return
        key = (w_norm, l_norm)
        if key in seen:
            return
        seen.add(key)
        variants.append({"winner": w_name, "loser": l_name, "tag": tag})

    add_variant(winner, loser, "winner_loser")

    players_distinct = (
        normalize_text(player_a) and normalize_text(player_b) and
        normalize_text(player_a) != normalize_text(player_b)
    )
    if players_distinct:
        if _names_equivalent(winner, player_a):
            add_variant(player_a, player_b, "player_fields_from_a")
        if _names_equivalent(winner, player_b):
            add_variant(player_b, player_a, "player_fields_from_b")
        if loser:
            if _names_equivalent(loser, player_a):
                add_variant(player_b, player_a, "player_fields_from_loser_a")
            if _names_equivalent(loser, player_b):
                add_variant(player_a, player_b, "player_fields_from_loser_b")

    return variants


def classify_upstream_row_issue(row):
    winner = _coalesce_name(row.get("winner"))
    loser = _coalesce_name(row.get("loser_name"))
    player_a = _coalesce_name(row.get("player_a"))
    player_b = _coalesce_name(row.get("player_b"))

    if not normalize_text(player_a) or not normalize_text(player_b):
        return True
    if normalize_text(player_a) == normalize_text(player_b):
        return True
    if winner and not (_names_equivalent(winner, player_a) or _names_equivalent(winner, player_b)):
        return True
    if loser and normalize_text(winner) == normalize_text(loser):
        return True
    if loser and not (_names_equivalent(loser, player_a) or _names_equivalent(loser, player_b)):
        return True
    return False


def row_name_index_tokens(name):
    name = _coalesce_name(name)
    if not name:
        return set()

    name_norm = normalize_text(name)
    tokens = set()
    if name_norm:
        tokens.add(name_norm)

    surname = _extract_match_surname(name)
    if surname:
        tokens.add(normalize_text(surname))

    abbrev_surname, _ = _parse_abbreviated_name(name)
    if abbrev_surname:
        tokens.add(abbrev_surname)
        abbrev_tokens = abbrev_surname.split()
        if len(abbrev_tokens) > 1:
            tokens.add(abbrev_tokens[0])

    alias = PLAYER_NAME_ALIASES.get(name_norm)
    if alias:
        tokens.add(normalize_text(alias))

    for alt in PLAYER_ALTERNATE_SURNAMES.get(name_norm, []):
        tokens.add(normalize_text(alt))

    return {token for token in tokens if token}


def source_name_index_tokens(name):
    name = _coalesce_name(name)
    if not name:
        return set()

    tokens = set()
    norm = normalize_text(name)
    if norm:
        tokens.add(norm)

    surname, _ = parse_odds_name(name)
    if surname:
        tokens.add(surname)

    return {token for token in tokens if token}


def build_pair_keys(left_tokens, right_tokens):
    keys = set()
    for left in left_tokens:
        for right in right_tokens:
            if not left or not right or left == right:
                continue
            keys.add(tuple(sorted((left, right))))
    return keys


def normalize_source_frame(frame, label, *, default_tour="atp"):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[
            "Date", "Winner", "Loser", "be_odds_w", "be_odds_l",
            "tournament", "tour", "source",
        ])

    df = frame.copy()
    if "Date" not in df.columns:
        return pd.DataFrame(columns=[
            "Date", "Winner", "Loser", "be_odds_w", "be_odds_l",
            "tournament", "tour", "source",
        ])

    dayfirst = label == "masterscup_local"
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=dayfirst)
    df = df.dropna(subset=["Date"]).copy()

    if "odds_w" in df.columns and "odds_l" in df.columns:
        df = df.rename(columns={"odds_w": "be_odds_w", "odds_l": "be_odds_l"})
    elif "B365W" in df.columns and "B365L" in df.columns:
        df = df.rename(columns={"B365W": "be_odds_w", "B365L": "be_odds_l"})
    elif "be_odds_w" in df.columns and "be_odds_l" in df.columns:
        pass
    else:
        df["be_odds_w"] = np.nan
        df["be_odds_l"] = np.nan

    if "Tournament" in df.columns and "tournament" not in df.columns:
        df["tournament"] = df["Tournament"]
    if "tournament" not in df.columns:
        df["tournament"] = label

    if "Round" in df.columns and "round" not in df.columns:
        df["round"] = df["Round"]

    if "tour" not in df.columns:
        df["tour"] = default_tour

    if "source" not in df.columns:
        df["source"] = label

    keep_cols = [
        "Date", "Winner", "Loser", "be_odds_w", "be_odds_l",
        "tournament", "tour", "source",
    ]
    for optional_col in ("round", "source_url", "source_note", "event_type"):
        if optional_col in df.columns:
            keep_cols.append(optional_col)

    df = df[keep_cols].copy()
    df["candidate_has_numeric_odds"] = df["be_odds_w"].notna() & df["be_odds_l"].notna()
    return df.reset_index(drop=True)


def build_source_candidate_table(frame, label, *, default_tour="atp"):
    df = normalize_source_frame(frame, label, default_tour=default_tour)
    pair_index = defaultdict(list)

    for idx, row in df.iterrows():
        pair_keys = build_pair_keys(
            source_name_index_tokens(row["Winner"]),
            source_name_index_tokens(row["Loser"]),
        )
        for pair_key in pair_keys:
            pair_index[(row["tour"], pair_key)].append(idx)

    return {"label": label, "df": df, "pair_index": pair_index}


def find_source_candidates(row, source_table, *, max_days=14, require_numeric=None):
    event_date = row.get("event_date")
    if pd.isna(event_date):
        return []

    variants = build_row_identity_variants(row)
    if not variants:
        return []

    pair_keys = set()
    for variant in variants:
        pair_keys |= build_pair_keys(
            row_name_index_tokens(variant["winner"]),
            row_name_index_tokens(variant["loser"]),
        )

    if not pair_keys:
        return []

    def match_candidates(candidates_df):
        if candidates_df.empty:
            return []

        matches = []
        candidates_df = candidates_df.sort_values(["Date", "Winner", "Loser"]).reset_index(drop=True)
        for _, candidate in candidates_df.iterrows():
            if not source_candidate_round_compatible(row, candidate):
                continue
            date_diff = abs((candidate["Date"] - event_date).days)
            for variant in variants:
                forward = (
                    names_match_flexible(variant["winner"], candidate["Winner"]) and
                    names_match_flexible(variant["loser"], candidate["Loser"])
                )
                reverse = (
                    names_match_flexible(variant["winner"], candidate["Loser"]) and
                    names_match_flexible(variant["loser"], candidate["Winner"])
                )
                if not forward and not reverse:
                    continue

                matches.append({
                    "candidate": candidate,
                    "swapped": bool(reverse),
                    "variant": variant,
                    "date_diff": int(date_diff),
                    "source_label": source_table["label"],
                })
                break
        return matches

    def filter_candidates(candidates_df):
        filtered = candidates_df[(candidates_df["Date"] - event_date).abs().dt.days <= max_days].copy()
        if require_numeric is True:
            filtered = filtered[filtered["candidate_has_numeric_odds"]]
        elif require_numeric is False:
            filtered = filtered[~filtered["candidate_has_numeric_odds"]]
        return filtered

    candidate_indices = set()
    for pair_key in pair_keys:
        candidate_indices.update(source_table["pair_index"].get((row["tour"], pair_key), []))

    keyed_candidates = source_table["df"].loc[sorted(candidate_indices)].copy() if candidate_indices else source_table["df"].iloc[0:0].copy()
    keyed_matches = match_candidates(filter_candidates(keyed_candidates))
    if keyed_matches:
        matches = keyed_matches
    else:
        fallback_candidates = source_table["df"][source_table["df"]["tour"] == row["tour"]].copy()
        matches = match_candidates(filter_candidates(fallback_candidates))

    matches.sort(key=lambda item: (
        item["date_diff"],
        str(item["candidate"]["Date"]),
        str(item["candidate"]["Winner"]),
        str(item["candidate"]["Loser"]),
        item["source_label"],
    ))
    return matches


def build_unmatched_diagnostics(still_unmatched_df, source_tables):
    rows = []
    for _, row in still_unmatched_df.iterrows():
        upstream_issue = classify_upstream_row_issue(row)
        best_numeric = None
        best_nonnumeric = None

        for source_table in source_tables:
            matches = find_source_candidates(row, source_table, max_days=14, require_numeric=None)
            matches = [
                match for match in matches
                if source_candidate_tournament_compatible(
                    row,
                    source_table["label"],
                    match["candidate"],
                )
            ]
            if not matches:
                continue
            match = matches[0]
            if match["candidate"]["candidate_has_numeric_odds"]:
                if best_numeric is None:
                    best_numeric = match
            elif best_nonnumeric is None:
                best_nonnumeric = match

        best_match = best_numeric or best_nonnumeric
        if upstream_issue:
            classification = "upstream_match_record_issue"
        elif best_numeric is not None:
            classification = "recoverable_matcher_gap"
        elif best_nonnumeric is not None:
            classification = "source_has_no_numeric_odds"
        else:
            classification = "source_absent"

        candidate = best_match["candidate"] if best_match is not None else None
        rows.append({
            "event_date": row["event_date"].date().isoformat() if pd.notna(row["event_date"]) else "",
            "tour": row["tour"],
            "tourney_name": row["tourney_name"],
            "bucket": f"{row['tourney_name']} {int(row['event_date'].year)}" if pd.notna(row["event_date"]) else str(row["tourney_name"]),
            "round": row.get("round", ""),
            "winner": row["winner"],
            "loser_name": row["loser_name"],
            "player_a": row.get("player_a", ""),
            "player_b": row.get("player_b", ""),
            "candidate_source": best_match["source_label"] if best_match is not None else "",
            "candidate_date": candidate["Date"].date().isoformat() if candidate is not None and pd.notna(candidate["Date"]) else "",
            "candidate_names": f"{candidate['Winner']} vs {candidate['Loser']}" if candidate is not None else "",
            "candidate_tournament": str(candidate["tournament"]) if candidate is not None else "",
            "candidate_has_numeric_odds": bool(candidate["candidate_has_numeric_odds"]) if candidate is not None else False,
            "classification": classification,
        })

    diagnostics_df = pd.DataFrame(rows).sort_values([
        "event_date", "tour", "tourney_name", "winner", "loser_name",
    ]).reset_index(drop=True)
    return diagnostics_df


def build_modeling_safe_exports(matches_df):
    suspected_reason_by_key = {
        make_match_row_key(
            row["event_date"],
            row["tour"],
            row["tourney_name"],
            row["round"],
            row["winner"],
            row["loser_name"],
        ): row["reason"]
        for row in SUSPECTED_UPSTREAM_BAD_MATCHES
    }
    export_columns = list(matches_df.columns) + ["quarantine_type", "quarantine_reason"]
    confirmed_records = []
    suspected_records = []
    confirmed_indices = set()
    suspected_indices = set()

    for idx, row in matches_df.iterrows():
        winner_norm = normalize_text(row.get("winner"))
        loser_norm = normalize_text(row.get("loser_name"))
        player_a_norm = normalize_text(row.get("player_a"))
        player_b_norm = normalize_text(row.get("player_b"))
        confirmed_reasons = []

        if winner_norm and loser_norm and winner_norm == loser_norm:
            confirmed_reasons.append("winner_equals_loser")
        if player_a_norm and player_b_norm and player_a_norm == player_b_norm:
            confirmed_reasons.append("player_a_equals_player_b")

        if confirmed_reasons:
            record = row.to_dict()
            record["quarantine_type"] = "confirmed_bad"
            record["quarantine_reason"] = ";".join(confirmed_reasons)
            confirmed_records.append(record)
            confirmed_indices.add(idx)
            continue

        row_key = make_match_row_key(
            row.get("event_date"),
            row.get("tour"),
            row.get("tourney_name"),
            row.get("round"),
            row.get("winner"),
            row.get("loser_name"),
        )
        suspected_reason = suspected_reason_by_key.get(row_key)
        if suspected_reason:
            record = row.to_dict()
            record["quarantine_type"] = "suspected_bad"
            record["quarantine_reason"] = suspected_reason
            suspected_records.append(record)
            suspected_indices.add(idx)

    excluded_indices = confirmed_indices | suspected_indices
    clean_df = matches_df.loc[~matches_df.index.isin(excluded_indices)].copy()
    confirmed_df = pd.DataFrame(confirmed_records, columns=export_columns)
    suspected_df = pd.DataFrame(suspected_records, columns=export_columns)
    return clean_df, confirmed_df, suspected_df


def main():
    print("Loading data...")
    odds_df = pd.read_csv(ODDS_PATH, low_memory=False)

    # Start from base matches.csv (no odds) and build matches_with_odds from scratch
    # Only process 2022+ rows which are the ones odds covers
    base_df = pd.read_csv(MATCHES_BASE_PATH, low_memory=False)
    base_df["event_date"] = pd.to_datetime(base_df["event_date"])

    # Split: pre-2022 (no odds possible) and 2022+ (odds possible)
    matches_df = base_df[base_df["event_date"] >= "2022-01-01"].copy().reset_index(drop=True)
    pre_2022_df = base_df[base_df["event_date"] < "2022-01-01"].copy()
    print(f"Loaded {len(odds_df)} odds rows, {len(base_df)} total match rows")
    print(f"  Pre-2022 rows (no odds): {len(pre_2022_df)}")
    print(f"  2022+ rows to match: {len(matches_df)}")

    # Add odds columns initialized to NaN
    for col in ["b365_a", "b365_b", "ps_a", "ps_b", "max_a", "max_b", "avg_a", "avg_b", "odds_date_diff"]:
        matches_df[col] = np.nan

    matches_df["year"] = matches_df["event_date"].dt.year

    # Normalize dates
    odds_df["Date"] = pd.to_datetime(odds_df["Date"])

    # Add normalized tournament name to odds
    odds_df["Tournament_norm"] = odds_df["Tournament"].map(normalize_text)

    # Build tournament lookup
    odds_lookup = build_tournament_lookup(odds_df)

    # ALL 2022+ rows need matching
    unmatched_indices = matches_df.index.tolist()
    print(f"Processing {len(unmatched_indices)} rows for odds matching")

    # Pre-filter: exclude tournaments that are definitely not in odds coverage
    excluded_patterns = [
        "125", "challenger", "davis cup", "bjk cup", "united cup", "atp cup",
        "billie jean king", "laver cup", "olympics", "olympic", "next gen",
        "nextgen", "wta elite", "playoff",
    ]

    # Track results
    matched_count = 0
    fail_reasons = defaultdict(int)
    fail_details = defaultdict(list)
    tournament_cache = {}
    newly_matched = {}

    # Pre-index odds by (tour, year, tournament_norm) for fast access
    odds_by_tourney = {}
    for (tour, year), group in odds_df.groupby(["tour", "source_year"]):
        for tnorm, tgroup in group.groupby("Tournament_norm"):
            odds_by_tourney[(tour, int(year), tnorm)] = tgroup

    # Also index ALL odds by (tour, year) for tournament-agnostic fallback
    odds_by_tour_year = {}
    for (tour, year), group in odds_df.groupby(["tour", "source_year"]):
        odds_by_tour_year[(tour, int(year))] = group

    total = len(unmatched_indices)
    skipped = 0
    attempted = 0
    fallback_matched = 0

    for i, idx in enumerate(unmatched_indices):
        if (i + 1) % 2000 == 0 or i == 0:
            print(f"Processing {i+1}/{total} (matched: {matched_count}, fallback: {fallback_matched})...")

        row = matches_df.loc[idx]
        tour = row["tour"]
        event_date = row["event_date"]
        if pd.isna(event_date):
            fail_reasons["no_date"] += 1
            continue

        year = int(event_date.year)
        our_tourney = str(row["tourney_name"])
        our_tourney_norm = normalize_text(our_tourney)

        # Quick exclude check
        if any(p in our_tourney_norm for p in excluded_patterns):
            fail_reasons["tournament_excluded"] += 1
            skipped += 1
            continue

        # Find odds tournament
        cache_key = (tour, year, our_tourney)
        if cache_key in tournament_cache:
            odds_tourney_norm = tournament_cache[cache_key]
        else:
            odds_tourney = find_odds_tournament(our_tourney, tour, year, odds_lookup)
            if odds_tourney is not None:
                odds_tourney_norm = normalize_text(odds_tourney)
            else:
                odds_tourney_norm = None
            tournament_cache[cache_key] = odds_tourney_norm

        use_fallback = False
        if odds_tourney_norm is None:
            # Tournament-agnostic fallback: search ALL odds for this (tour, year)
            use_fallback = True

        attempted += 1

        if use_fallback:
            tournament_odds = odds_by_tour_year.get((tour, year), pd.DataFrame())
            if tournament_odds.empty:
                fail_reasons["no_odds_for_tour_year"] += 1
                continue
        else:
            odds_key = (tour, year, odds_tourney_norm)
            tournament_odds = odds_by_tourney.get(odds_key, pd.DataFrame())

        # For year-boundary matches (late December), also check next year's odds
        if event_date.month == 12 and event_date.day >= 25:
            if use_fallback:
                next_year_odds = odds_by_tour_year.get((tour, year + 1), pd.DataFrame())
            else:
                next_year_odds = odds_by_tourney.get((tour, year + 1, odds_tourney_norm), pd.DataFrame())
            if not next_year_odds.empty:
                tournament_odds = pd.concat([tournament_odds, next_year_odds], ignore_index=True)

        # Determine date window - use wider window for fallback (no tournament filter)
        grand_slams = {"australian open", "french open", "wimbledon", "us open", "roland garros"}
        if use_fallback:
            max_days = 21  # wider window since no tournament constraint
        else:
            max_days = 21 if our_tourney_norm in grand_slams or odds_tourney_norm in grand_slams else 14

        odds_row, method = try_match_row(row, tournament_odds, event_date, max_days)

        if odds_row is not None:
            odds_dict = assign_odds_to_row(row, odds_row, method)
            date_diff = abs((odds_row["Date"] - event_date).days)
            odds_dict["odds_date_diff"] = float(date_diff)

            # Validate we got at least some actual odds numbers
            has_any_odds = any(
                not pd.isna(odds_dict.get(col))
                for col in ["b365_a", "ps_a", "max_a", "avg_a"]
            )
            if has_any_odds:
                newly_matched[idx] = odds_dict
                matched_count += 1
                if use_fallback:
                    fallback_matched += 1
            else:
                fail_reasons["odds_values_missing"] += 1
        else:
            reason_key = f"fallback_{method}" if use_fallback else method
            fail_reasons[reason_key] += 1
            if method == "name_mismatch":
                winner = row["winner"]
                loser = row["loser_name"]
                fail_details["name_mismatch"].append(
                    f"{tour} {event_date.date()}: {winner} vs {loser} @ {our_tourney}"
                )
            elif method == "no_odds_in_date_range":
                fail_details["no_odds_in_date_range"].append(
                    f"{tour} {event_date.date()}: @ {our_tourney}"
                )

    print(f"\nProcessing complete.")
    print(f"  Total rows: {total}")
    print(f"  Skipped (excluded events): {skipped}")
    print(f"  Attempted: {attempted}")
    print(f"  Matched: {matched_count} (of which {fallback_matched} via tournament-agnostic fallback)")

    # Apply the new matches
    for idx, odds_dict in newly_matched.items():
        for col, val in odds_dict.items():
            matches_df.at[idx, col] = val

    # ========================
    # PASS 2: BetExplorer odds for excluded events (WTA 125, team events, etc.)
    # ========================
    betexplorer_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_odds.csv"
    betexplorer_wta125_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_wta125_odds.csv"
    betexplorer_team_events_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv"
    if betexplorer_path.exists() or betexplorer_wta125_path.exists() or betexplorer_team_events_path.exists():
        print("\n--- Pass 2: BetExplorer odds for excluded events ---")
        be_parts = []

        # Load main BetExplorer file (odds_w/odds_l columns)
        if betexplorer_path.exists():
            be1 = pd.read_csv(betexplorer_path)
            be1["Date"] = pd.to_datetime(be1["Date"], errors="coerce")
            be1 = be1.dropna(subset=["Date"])
            be1 = be1[be1["odds_w"].notna() & be1["odds_l"].notna()].copy()
            # Rename to common format
            be1 = be1.rename(columns={"odds_w": "be_odds_w", "odds_l": "be_odds_l"})
            be1_cols = ["Date", "Winner", "Loser", "be_odds_w", "be_odds_l", "tournament", "tour"]
            if "round" in be1.columns:
                be1_cols.append("round")
            be_parts.append(be1[be1_cols])

        # Load WTA 125 file (B365W/B365L columns)
        if betexplorer_wta125_path.exists():
            be2 = pd.read_csv(betexplorer_wta125_path)
            be2["Date"] = pd.to_datetime(be2["Date"], errors="coerce")
            be2 = be2.dropna(subset=["Date"])
            be2 = be2[be2["B365W"].notna() & be2["B365L"].notna()].copy()
            be2 = be2.rename(columns={"B365W": "be_odds_w", "B365L": "be_odds_l"})
            if "tour" not in be2.columns:
                be2["tour"] = "wta"
            be2_cols = ["Date", "Winner", "Loser", "be_odds_w", "be_odds_l", "tournament", "tour"]
            if "round" in be2.columns:
                be2_cols.append("round")
            be_parts.append(be2[be2_cols])

        # The targeted BetExplorer scraper now emits additional WTA 125 rows into the
        # team events file. Feed those rows through the stronger excluded-event matcher
        # instead of leaving them to the generic Pass 3 date/name fallback.
        if betexplorer_team_events_path.exists():
            be3 = pd.read_csv(betexplorer_team_events_path)
            be3["Date"] = pd.to_datetime(be3["Date"], errors="coerce")
            be3 = be3.dropna(subset=["Date"])
            if "event_type" in be3.columns:
                be3 = be3[be3["event_type"].astype(str) == "125"].copy()
            else:
                be3 = be3.iloc[0:0].copy()
            if not be3.empty:
                be3 = be3[be3["odds_w"].notna() & be3["odds_l"].notna()].copy()
                be3 = be3.rename(columns={"odds_w": "be_odds_w", "odds_l": "be_odds_l"})
                if "tour" not in be3.columns:
                    be3["tour"] = "wta"
                be3_cols = ["Date", "Winner", "Loser", "be_odds_w", "be_odds_l", "tournament", "tour"]
                if "round" in be3.columns:
                    be3_cols.append("round")
                be_parts.append(be3[be3_cols])

        if not be_parts:
            print("  No BetExplorer rows with both odds found for excluded events")
            be_with_odds = pd.DataFrame(columns=["Date", "Winner", "Loser", "be_odds_w", "be_odds_l", "tournament", "tour"])
        else:
            be_with_odds = pd.concat(be_parts, ignore_index=True)
        # Deduplicate: prefer rows from wta125 file (more complete odds)
        be_with_odds = be_with_odds.drop_duplicates(
            subset=["Date", "Winner", "Loser"], keep="last"
        )
        print(f"  BetExplorer rows with both odds: {len(be_with_odds)}")

        # Find excluded matches that still have no odds
        still_no_odds = matches_df[~has_any_odds_mask(matches_df)].copy()
        excluded_no_odds = still_no_odds[
            still_no_odds["tourney_name"].map(
                lambda t: any(p in normalize_text(t) for p in excluded_patterns)
            )
        ]
        print(f"  Excluded events without odds: {len(excluded_no_odds)}")

        # Index BetExplorer by tournament for faster lookup
        be_with_odds["tournament_norm"] = be_with_odds["tournament"].map(normalize_text)

        # Build tournament name mapping: match name -> BetExplorer name
        # e.g. "Angers 125" -> "angers", "Antalya 125 #2" -> "antalya-2"
        def match_tourney_to_be(tourney_name):
            """Map match tournament name to BetExplorer tournament name."""
            norm = normalize_text(tourney_name)
            # Remove "125", "#N" suffixes, clean up
            norm = re.sub(r'\b125\b', '', norm).strip()
            # Handle "#2" -> "-2" style
            num_match = re.search(r'#(\d+)', tourney_name)
            suffix = f"-{num_match.group(1)}" if num_match else ""
            # Base name: first word(s) before numbers
            base = re.sub(r'\d+', '', norm).strip()
            base = base.replace(' ', '-') if ' ' in base else base
            return base + suffix if suffix else base

        be_matched = 0
        be_details = defaultdict(int)

        for idx, row in excluded_no_odds.iterrows():
            event_date = row["event_date"]
            if pd.isna(event_date):
                continue
            winner = row["winner"]
            loser = row["loser_name"]
            player_a = row["player_a"]
            tour = row["tour"]
            a_is_winner = _names_equivalent(player_a, winner)
            row_variants = build_row_identity_variants(row)
            if not row_variants:
                be_details["name_mismatch"] += 1
                continue

            # Filter BetExplorer by tour first
            tour_mask = be_with_odds["tour"] == tour
            tour_filtered = be_with_odds[tour_mask]
            if tour_filtered.empty:
                be_details["no_tour_match"] += 1
                continue

            # Filter by date proximity
            date_mask = (tour_filtered["Date"] - event_date).abs().dt.days <= 7
            candidates = tour_filtered[date_mask]
            if candidates.empty:
                be_details["no_date_match"] += 1
                continue

            # Try tournament-level filtering first for tighter matching, but fall back
            # to the full date candidate pool if generic labels ("Mumbai") hide the
            # year-suffixed rows that actually contain the right players.
            be_tourney = match_tourney_to_be(str(row["tourney_name"]))
            be_tourney_norm = normalize_text(be_tourney)
            be_tourney_base = be_tourney.split("-")[0]
            candidate_groups = []

            exact_candidates = candidates[candidates["tournament_norm"] == be_tourney_norm]
            if not exact_candidates.empty:
                candidate_groups.append(exact_candidates)

            partial_candidates = candidates[
                candidates["tournament_norm"].str.contains(be_tourney_base, na=False)
            ]
            if not partial_candidates.empty:
                candidate_groups.append(partial_candidates)

            candidate_groups.append(candidates)

            matched_row = None
            swapped = False
            seen_candidate_keys = set()
            for candidate_group in candidate_groups:
                for _, be_row in candidate_group.iterrows():
                    candidate_key = (
                        be_row["Date"],
                        be_row["Winner"],
                        be_row["Loser"],
                        be_row.get("tournament", ""),
                    )
                    if candidate_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(candidate_key)
                    if not source_candidate_round_compatible(row, be_row):
                        continue

                    for variant in row_variants:
                        w_match = names_match_flexible(variant["winner"], be_row["Winner"])
                        l_match = names_match_flexible(variant["loser"], be_row["Loser"])
                        if w_match and l_match:
                            matched_row = be_row
                            swapped = False
                            break
                        w_rev = names_match_flexible(variant["winner"], be_row["Loser"])
                        l_rev = names_match_flexible(variant["loser"], be_row["Winner"])
                        if w_rev and l_rev:
                            matched_row = be_row
                            swapped = True
                            break
                    if matched_row is not None:
                        break
                if matched_row is not None:
                    break

            if matched_row is None:
                be_details["name_mismatch"] += 1
                continue

            # Assign odds: be_odds_w = winner odds, be_odds_l = loser odds
            odds_w = float(matched_row["be_odds_w"])
            odds_l = float(matched_row["be_odds_l"])

            if swapped:
                # BetExplorer Winner/Loser are swapped relative to our data
                odds_w, odds_l = odds_l, odds_w

            if a_is_winner:
                matches_df.at[idx, "b365_a"] = odds_w
                matches_df.at[idx, "b365_b"] = odds_l
            else:
                matches_df.at[idx, "b365_a"] = odds_l
                matches_df.at[idx, "b365_b"] = odds_w

            date_diff = abs((matched_row["Date"] - event_date).days)
            matches_df.at[idx, "odds_date_diff"] = float(date_diff)
            be_matched += 1

        print(f"  BetExplorer matched: {be_matched}")
        for reason, count in sorted(be_details.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        fail_reasons["betexplorer_no_date"] = be_details.get("no_date_match", 0)
        fail_reasons["betexplorer_name_mismatch"] = be_details.get("name_mismatch", 0)
    else:
        print("\n--- No BetExplorer data found, skipping Pass 2 ---")

    # ========================
    # PASS 3: Additional odds sources (OddsPortal, Flashscore, extra BetExplorer)
    # ========================
    source_specs = [
        ("betexplorer_team_events", ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv", "atp"),
        ("oddsportal_team_events", ROOT / "data" / "raw" / "tennis" / "oddsportal_team_events_odds.csv", "atp"),
        ("betexplorer_wta125", ROOT / "data" / "raw" / "tennis" / "betexplorer_wta125_odds.csv", "wta"),
        ("oddsportal_wta125", ROOT / "data" / "raw" / "tennis" / "oddsportal_wta125_odds.csv", "wta"),
        ("flashscore", ROOT / "data" / "raw" / "tennis" / "flashscore_odds.csv", "atp"),
        ("betexplorer_extra", ROOT / "data" / "raw" / "tennis" / "betexplorer_extra_odds.csv", "atp"),
        ("welcome_bet", ROOT / "data" / "raw" / "tennis" / "welcome_bet_odds.csv", "atp"),
        ("manual_search_snippets", ROOT / "data" / "raw" / "tennis" / "manual_search_snippets_odds.csv", "wta"),
        ("manual_search_articles", ROOT / "data" / "raw" / "tennis" / "manual_search_articles_odds.csv", "wta"),
        ("signalodds", ROOT / "data" / "raw" / "tennis" / "signalodds_odds.csv", "wta"),
        ("wayback_tennis", ROOT / "data" / "raw" / "tennis" / "wayback_tennis_odds.csv", "atp"),
        ("masterscup_local", ROOT / "data" / "raw" / "tennis" / "masterscup_odds.csv", "atp"),
    ]
    extra_source_tables = []
    diagnostic_source_tables = [build_source_candidate_table(odds_df, "tennis_data")]

    for label, src_path, default_tour in source_specs:
        if not src_path.exists():
            continue
        try:
            raw_df = pd.read_csv(src_path, low_memory=False)
            source_table = build_source_candidate_table(raw_df, label, default_tour=default_tour)
            if source_table["df"].empty:
                continue

            numeric_df = source_table["df"][source_table["df"]["candidate_has_numeric_odds"]].copy()
            if not numeric_df.empty:
                extra_source_tables.append({
                    "label": label,
                    "df": numeric_df.reset_index(drop=True),
                    "pair_index": build_source_candidate_table(
                        numeric_df,
                        label,
                        default_tour=default_tour,
                    )["pair_index"],
                })
                print(f"  Loaded {len(numeric_df)} rows from {src_path.name}")

            diagnostic_source_tables.append(source_table)
        except Exception as e:
            print(f"  Error loading {src_path.name}: {e}")

    if extra_source_tables:
        print(f"\n--- Pass 3: Additional odds sources ---")
        total_extra_rows = sum(len(table["df"]) for table in extra_source_tables)
        print(f"  Extra rows with odds: {total_extra_rows}")

        still_no_odds = matches_df[~has_any_odds_mask(matches_df)].copy()
        print(f"  Matches without any odds: {len(still_no_odds)}")

        extra_matched = 0
        for idx, row in still_no_odds.iterrows():
            event_date = row["event_date"]
            if pd.isna(event_date):
                continue

            a_is_winner = _names_equivalent(row["player_a"], row["winner"])
            matched_candidate = None
            matched_source = None

            for source_table in extra_source_tables:
                matches = find_source_candidates(row, source_table, max_days=14, require_numeric=True)
                matches = [
                    match for match in matches
                    if source_candidate_tournament_compatible(
                        row,
                        source_table["label"],
                        match["candidate"],
                    )
                ]
                if matches:
                    matched_candidate = matches[0]
                    matched_source = source_table["label"]
                    break

            if matched_candidate is None:
                continue

            matched_row = matched_candidate["candidate"]
            odds_w = float(matched_row["be_odds_w"])
            odds_l = float(matched_row["be_odds_l"])
            if matched_candidate["swapped"]:
                odds_w, odds_l = odds_l, odds_w
            if a_is_winner:
                matches_df.at[idx, "b365_a"] = odds_w
                matches_df.at[idx, "b365_b"] = odds_l
            else:
                matches_df.at[idx, "b365_a"] = odds_l
                matches_df.at[idx, "b365_b"] = odds_w
            date_diff = abs((matched_row["Date"] - event_date).days)
            matches_df.at[idx, "odds_date_diff"] = float(date_diff)
            extra_matched += 1

        print(f"  Extra sources matched: {extra_matched}")
    else:
        diagnostic_source_tables = [build_source_candidate_table(odds_df, "tennis_data")]

    # Summary stats - use ANY odds column, not just b365_a
    any_odds_mask = has_any_odds_mask(matches_df)
    total_with_odds = int(any_odds_mask.sum())
    still_unmatched_df = matches_df[~any_odds_mask].copy()
    still_unmatched_count = len(still_unmatched_df)

    diagnostics_df = build_unmatched_diagnostics(still_unmatched_df, diagnostic_source_tables)
    diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)
    diagnostic_counts = diagnostics_df["classification"].value_counts().to_dict() if not diagnostics_df.empty else {}
    modeling_safe_df, confirmed_bad_df, suspected_bad_df = build_modeling_safe_exports(matches_df)
    base_modeling_safe_df, base_confirmed_bad_df, base_suspected_bad_df = build_modeling_safe_exports(base_df)
    modeling_safe_with_odds = int(has_any_odds_mask(modeling_safe_df).sum())

    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TENNIS ODDS MATCHING IMPROVEMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"")
    report_lines.append(f"Total rows: {total}")
    report_lines.append(f"Excluded (125s, team events, Olympics, etc.): {skipped}")
    report_lines.append(f"Attempted to match: {attempted}")
    report_lines.append(f"Matched with odds: {matched_count} ({fallback_matched} via fallback)")
    report_lines.append(f"Unmatched (all): {still_unmatched_count}")
    report_lines.append(f"Total rows with odds: {total_with_odds}")
    report_lines.append(f"")
    report_lines.append(f"--- Unmatched Classification ---")
    for reason, count in sorted(diagnostic_counts.items(), key=lambda x: (-x[1], x[0])):
        report_lines.append(f"  {reason}: {count}")
    report_lines.append(f"")
    report_lines.append(f"--- Modeling Safe Exclusions ---")
    report_lines.append(f"  confirmed_bad: {len(confirmed_bad_df)}")
    report_lines.append(f"  suspected_bad: {len(suspected_bad_df)}")
    report_lines.append(f"  modeling_safe_rows: {len(modeling_safe_df)}")
    report_lines.append(f"  modeling_safe_with_odds: {modeling_safe_with_odds}")
    report_lines.append(f"")
    report_lines.append(f"--- Base Match Table Safe Exclusions ---")
    report_lines.append(f"  confirmed_bad: {len(base_confirmed_bad_df)}")
    report_lines.append(f"  suspected_bad: {len(base_suspected_bad_df)}")
    report_lines.append(f"  base_modeling_safe_rows: {len(base_modeling_safe_df)}")
    report_lines.append(f"")
    report_lines.append(f"--- Failure Breakdown ---")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {reason}: {count}")

    report_lines.append(f"")
    report_lines.append(f"--- Tournament Not In Odds (unique) ---")
    unique_not_in_odds = sorted(set(fail_details.get("tournament_not_in_odds", [])))
    for item in unique_not_in_odds[:50]:
        report_lines.append(f"  {item}")

    report_lines.append(f"")
    report_lines.append(f"--- No Odds In Date Range (unique, first 30) ---")
    unique_date_range = sorted(set(fail_details.get("no_odds_in_date_range", [])))
    for item in unique_date_range[:30]:
        report_lines.append(f"  {item}")

    report_lines.append(f"")
    report_lines.append(f"--- Name Mismatch Details (first 50) ---")
    for item in fail_details.get("name_mismatch", [])[:50]:
        report_lines.append(f"  {item}")

    report_lines.append(f"")
    report_lines.append(f"--- Coverage by Year ---")
    for year in sorted(matches_df["event_date"].dt.year.dropna().unique()):
        year_df = matches_df[matches_df["event_date"].dt.year == year]
        year_has_odds = (
            year_df["b365_a"].notna() | year_df["ps_a"].notna() |
            year_df["max_a"].notna() | year_df["avg_a"].notna()
        )
        with_odds = year_has_odds.sum()
        total_year = len(year_df)
        pct = 100.0 * with_odds / total_year if total_year > 0 else 0
        report_lines.append(f"  {int(year)}: {with_odds}/{total_year} ({pct:.1f}%)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")
    print(f"Diagnostics saved to {DIAGNOSTICS_PATH}")

    # Save only the 2022+ rows (matching original matches_with_odds.csv scope)
    matches_df.to_csv(OUTPUT_PATH, index=False)
    base_modeling_safe_df.to_csv(MATCHES_BASE_MODELING_SAFE_PATH, index=False)
    modeling_safe_df.to_csv(MODELING_SAFE_OUTPUT_PATH, index=False)
    confirmed_bad_df.to_csv(CONFIRMED_BAD_ROWS_PATH, index=False)
    suspected_bad_df.to_csv(SUSPECTED_BAD_ROWS_PATH, index=False)
    base_confirmed_bad_df.to_csv(BASE_CONFIRMED_BAD_ROWS_PATH, index=False)
    base_suspected_bad_df.to_csv(BASE_SUSPECTED_BAD_ROWS_PATH, index=False)
    print(f"Saved {len(matches_df)} rows to {OUTPUT_PATH}")
    has_any = has_any_odds_mask(matches_df)
    print(f"  With any odds: {int(has_any.sum())}")
    print(f"  With B365 odds: {int(matches_df['b365_a'].notna().sum())}")
    print(f"Saved {len(base_modeling_safe_df)} rows to {MATCHES_BASE_MODELING_SAFE_PATH}")
    print(f"Saved {len(base_confirmed_bad_df)} confirmed bad base rows to {BASE_CONFIRMED_BAD_ROWS_PATH}")
    print(f"Saved {len(base_suspected_bad_df)} suspected bad base rows to {BASE_SUSPECTED_BAD_ROWS_PATH}")
    print(f"Saved {len(modeling_safe_df)} rows to {MODELING_SAFE_OUTPUT_PATH}")
    print(f"  Modeling-safe rows with any odds: {modeling_safe_with_odds}")
    print(f"Saved {len(confirmed_bad_df)} confirmed bad rows to {CONFIRMED_BAD_ROWS_PATH}")
    print(f"Saved {len(suspected_bad_df)} suspected bad rows to {SUSPECTED_BAD_ROWS_PATH}")


if __name__ == "__main__":
    main()
