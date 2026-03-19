"""
Curated V5 web backfill for the remaining recoverable gaps.

This script appends:
1. UFC Shanghai moneylines recovered from public opening-odds coverage
2. Method-of-victory rows recovered from indexed sportsbook/article pages

It is intentionally idempotent:
- it skips fights already present in the master CSVs
- it rewrites the missing-row CSVs from the master data after appending
"""

from __future__ import annotations

import csv
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ML_PATH = REPO / "data" / "raw" / "historical_odds" / "historical_odds.csv"
METHOD_PATH = REPO / "data" / "raw" / "method_odds" / "historical_method_odds_all.csv"
MISSING_ML_PATH = REPO / "data" / "raw" / "v5_missing_moneyline.csv"
STILL_MISSING_ML_PATH = REPO / "data" / "raw" / "v5_still_missing_moneyline.csv"
MISSING_METHOD_PATH = REPO / "data" / "raw" / "v5_missing_method_odds.csv"


def norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", text).lower().split())


def fight_key(event_date: str, fighter_a: str, fighter_b: str) -> str:
    names = sorted([norm(fighter_a), norm(fighter_b)])
    return f"{event_date}|{names[0]}|{names[1]}"


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / abs(odds)


def american_to_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_to_prob(odds: float) -> float:
    return 1.0 / odds


def fractional_to_prob(text: str) -> float:
    num, den = text.split("/")
    return 1.0 / (1.0 + (float(num) / float(den)))


def odds_value_to_prob(kind: str, value) -> float:
    if kind == "am":
        return american_to_prob(float(value))
    if kind == "dec":
        return decimal_to_prob(float(value))
    if kind == "frac":
        return fractional_to_prob(str(value))
    raise ValueError(f"Unsupported odds kind: {kind}")


MMAODDSBREAKER_SHANGHAI_URL = (
    "https://www.mmaoddsbreaker.com/fight-odds/opening-odds/"
    "160595-opening-betting-odds-for-ufc-shanghai-walker-vs-mingyang/"
)

RECOVERED_MONEYLINES = [
    {
        "event_date": "2024-03-23",
        "fighter_a": "Igor Severino",
        "fighter_b": "Andre Lima",
        "a_american": -185,
        "b_american": +150,
        "query_date": "2024-03-23",
        "source_url": "user_provided_2026-03-19",
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Maheshate",
        "fighter_b": "Gauge Young",
        "a_american": +130,
        "b_american": -150,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Rongzhu",
        "fighter_b": "Austin Hubbard",
        "a_american": -300,
        "b_american": +250,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Sergei Pavlovich",
        "fighter_b": "Waldo Cortes Acosta",
        "a_american": -350,
        "b_american": +285,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Sumudaerji",
        "fighter_b": "Kevin Borjas",
        "a_american": -170,
        "b_american": +145,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Xiao Long",
        "fighter_b": "SuYoung You",
        "a_american": +145,
        "b_american": -170,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
    {
        "event_date": "2025-08-23",
        "fighter_a": "Yizha",
        "fighter_b": "Westin Wilson",
        "a_american": -1400,
        "b_american": +800,
        "query_date": "2025-08-17",
        "source_url": MMAODDSBREAKER_SHANGHAI_URL,
    },
]


RECOVERED_METHOD_ROWS = [
    {
        "event_date": "2023-08-26",
        "fighter_a": "Waldo Cortes Acosta",
        "fighter_b": "Lukasz Brzeski",
        "source": "web_betmma_search_multi",
        "source_url": (
            "https://www.betmma.tips/handicapper_props.php?ID=139591&Prop=9 | "
            "https://www.betmma.tips/Spliffyk%26ShowAll%3D1"
        ),
        "markets": {
            "a_ko_odds_prob": ("dec", 3.50),
            "b_dec_odds_prob": ("dec", 5.50),
        },
    },
    {
        "event_date": "2023-11-11",
        "fighter_a": "Nazim Sadykhov",
        "fighter_b": "Viacheslav Borshchev",
        "source": "web_betmma_search_multi",
        "source_url": (
            "https://www.betmma.tips/ufc_betting_advice.php?Fight=11347&For="
            "Viacheslav_Borshchev_vs_Nazim_Sadykhov"
        ),
        "markets": {
            "a_sub_odds_prob": ("dec", 5.20),
            "a_dec_odds_prob": ("dec", 4.00),
            "b_ko_odds_prob": ("dec", 3.60),
            "b_dec_odds_prob": ("dec", 5.50),
        },
    },
    {
        "event_date": "2023-12-16",
        "fighter_a": "Martin Buday",
        "fighter_b": "Shamil Gaziev",
        "source": "web_betmma_search_multi",
        "source_url": (
            "https://www.betmma.tips/GuruScoutingMMA%26ShowAll%3D1 | "
            "https://www.betmma.tips/NewtBetsMMA"
        ),
        "markets": {
            "a_dec_odds_prob": ("dec", 3.55),
            "b_sub_odds_prob": ("dec", 10.00),
        },
    },
    {
        "event_date": "2024-02-03",
        "fighter_a": "Aliaskhab Khizriev",
        "fighter_b": "Makhmud Muradov",
        "source": "web_telecomasia_search",
        "source_url": "https://www.telecomasia.net/events/aliaskhab-khizriev-vs-makhmud-muradov-preview-where-to-watch-and-betting-odds/",
        "markets": {
            "a_sub_odds_prob": ("dec", 4.33),
            "b_dec_odds_prob": ("dec", 5.25),
        },
    },
    {
        "event_date": "2024-03-16",
        "fighter_a": "Bryan Battle",
        "fighter_b": "Ange Loosa",
        "source": "web_covers",
        "source_url": "https://www.covers.com/ufc/fight-night-battle-vs-loosa-odds-picks-predictions",
        "markets": {
            "a_ko_odds_prob": ("am", +405),
            "a_sub_odds_prob": ("am", +540),
            "a_dec_odds_prob": ("am", +160),
            "b_ko_odds_prob": ("am", +600),
            "b_sub_odds_prob": ("am", +1450),
            "b_dec_odds_prob": ("am", +320),
        },
    },
    {
        "event_date": "2024-03-23",
        "fighter_a": "Montse Rendon",
        "fighter_b": "Daria Zhelezniakova",
        "source": "web_betmma_search_multi",
        "source_url": (
            "https://www.betmma.tips/Mike_Rhodes | "
            "https://www.betmma.tips/AlexKane%26ShowAll%3D1"
        ),
        "markets": {
            "b_ko_odds_prob": ("dec", 4.75),
            "b_dec_odds_prob": ("dec", 2.25),
        },
    },
    {
        "event_date": "2024-06-01",
        "fighter_a": "Mickey Gall",
        "fighter_b": "Bassil Hafez",
        "source": "web_ringside24_search",
        "source_url": "https://ringside24.com/en/news/mma/146861-what-time-is-ufc-302-tonight-gall-vs-hafez-start-times-schedules-fight-card",
        "markets": {
            "a_ko_odds_prob": ("am", +1800),
            "a_dec_odds_prob": ("am", +800),
            "b_ko_odds_prob": ("am", +340),
            "b_dec_odds_prob": ("am", +175),
        },
    },
    {
        "event_date": "2024-08-24",
        "fighter_a": "Zach Reese",
        "fighter_b": "Jose Daniel Medina",
        "source": "web_betmma_search_multi",
        "source_url": (
            "https://www.betmma.tips/handicapper_props.php?ID=137279&Prop=10 | "
            "https://www.betmma.tips/ReachCombat"
        ),
        "markets": {
            "a_ko_odds_prob": ("dec", 2.25),
            "a_sub_odds_prob": ("dec", 2.80),
            "b_dec_odds_prob": ("dec", 10.00),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Calvin Kattar",
        "fighter_b": "Steve Garcia",
        "source": "web_covers",
        "source_url": "https://www.covers.com/ufc/fight-night-kattar-vs-garcia-predictions-picks-odds",
        "markets": {
            "a_ko_odds_prob": ("am", +215),
            "a_sub_odds_prob": ("am", +2400),
            "a_dec_odds_prob": ("am", +475),
            "b_ko_odds_prob": ("am", +250),
            "b_sub_odds_prob": ("am", +1900),
            "b_dec_odds_prob": ("am", +210),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Derrick Lewis",
        "fighter_b": "Tallison Teixeira",
        "source": "web_covers",
        "source_url": "https://www.covers.com/ufc/fight-night-lewis-vs-teixeira-predictions-picks-odds",
        "markets": {
            "a_ko_odds_prob": ("am", +300),
            "a_sub_odds_prob": ("am", +3300),
            "a_dec_odds_prob": ("am", +2000),
            "b_ko_odds_prob": ("am", -190),
            "b_sub_odds_prob": ("am", +900),
            "b_dec_odds_prob": ("am", +900),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Fatima Kline",
        "fighter_b": "Melissa Martinez",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.es/en/sports/mixed-martial-arts/26420387/ufc-matches/10581356/fatima-kline-v-melissa-martinez/34448891/",
        "markets": {
            "a_ko_odds_prob": ("dec", 3.60),
            "a_sub_odds_prob": ("dec", 4.00),
            "a_dec_odds_prob": ("dec", 2.10),
            "b_ko_odds_prob": ("dec", 23.00),
            "b_sub_odds_prob": ("dec", 31.00),
            "b_dec_odds_prob": ("dec", 15.00),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Jake Matthews",
        "fighter_b": "Chidi Njokuani",
        "source": "web_betus_search",
        "source_url": "https://www.betus.com.pa/sportsbook/martial-arts/props/jake-matthews-vs-chidi-njokuani/",
        "markets": {
            "a_ko_odds_prob": ("am", +700),
            "a_sub_odds_prob": ("am", +700),
            "a_dec_odds_prob": ("am", +250),
            "b_ko_odds_prob": ("am", +315),
            "b_sub_odds_prob": ("am", +2000),
            "b_dec_odds_prob": ("am", +140),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Junior Tafa",
        "fighter_b": "Tuco Tokkos",
        "source": "web_betus_search",
        "source_url": "https://www.betus.com.pa/sportsbook/martial-arts/props/tuco-tokkos-vs-junior-tafa/",
        "markets": {
            "a_ko_odds_prob": ("am", -110),
            "a_sub_odds_prob": ("am", +2500),
            "a_dec_odds_prob": ("am", +800),
            "b_ko_odds_prob": ("am", +335),
            "b_sub_odds_prob": ("am", +450),
            "b_dec_odds_prob": ("am", +800),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Kennedy Nzechukwu",
        "fighter_b": "Valter Walker",
        "source": "web_sportingbet_search",
        "source_url": "https://www.sportingbet.co.za/en/sports/events/kennedy-nzechukwu-ngr-valter-walker-bra-17710257",
        "markets": {
            "a_ko_odds_prob": ("dec", 2.50),
            "a_sub_odds_prob": ("dec", 9.00),
            "a_dec_odds_prob": ("dec", 3.50),
            "b_ko_odds_prob": ("dec", 9.00),
            "b_sub_odds_prob": ("dec", 7.50),
            "b_dec_odds_prob": ("dec", 5.00),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Lauren Murphy",
        "fighter_b": "Eduarda Moura",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.es/en/sports/mixed-martial-arts/26420387/ufc-matches/10581356/lauren-murphy-v-eduarda-moura/34448892/",
        "markets": {
            "a_ko_odds_prob": ("dec", 21.00),
            "a_sub_odds_prob": ("dec", 21.00),
            "a_dec_odds_prob": ("dec", 8.50),
            "b_ko_odds_prob": ("dec", 8.00),
            "b_sub_odds_prob": ("dec", 6.50),
            "b_dec_odds_prob": ("dec", 1.57),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Max Griffin",
        "fighter_b": "Chris Curtis",
        "source": "web_888sport_search",
        "source_url": "https://www.888sport.com/mma/world/ultimate-fighting-championship/max-griffin-v-chris-curtis-e-6015077/",
        "markets": {
            "a_ko_odds_prob": ("frac", "11/2"),
            "a_sub_odds_prob": ("frac", "25/1"),
            "a_dec_odds_prob": ("frac", "5/1"),
            "b_ko_odds_prob": ("frac", "3/1"),
            "b_sub_odds_prob": ("frac", "25/1"),
            "b_dec_odds_prob": ("frac", "46/63"),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Mitch Ramirez",
        "fighter_b": "Mike Davis",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.es/en/sports/mixed-martial-arts/26420387/ufc-matches/10581356/mike-davis-v-mitch-ramirez/34476493/",
        "markets": {
            "a_ko_odds_prob": ("dec", 19.00),
            "a_sub_odds_prob": ("dec", 36.00),
            "a_dec_odds_prob": ("dec", 13.00),
            "b_ko_odds_prob": ("dec", 2.40),
            "b_sub_odds_prob": ("dec", 3.75),
            "b_dec_odds_prob": ("dec", 3.20),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Nate Landwehr",
        "fighter_b": "Morgan Charriere",
        "source": "web_sportingbet_search",
        "source_url": "https://www.sportingbet.co.za/en/sports/events/nate-landwehr-usa-morgan-charriere-fra-17710262",
        "markets": {
            "a_ko_odds_prob": ("dec", 7.50),
            "a_sub_odds_prob": ("dec", 19.00),
            "a_dec_odds_prob": ("dec", 5.50),
            "b_ko_odds_prob": ("dec", 3.00),
            "b_sub_odds_prob": ("dec", 11.00),
            "b_dec_odds_prob": ("dec", 2.25),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Stephen Thompson",
        "fighter_b": "Gabriel Bonfim",
        "source": "web_covers",
        "source_url": "https://www.covers.com/ufc/fight-night-thompson-vs-bonfim-predictions-picks-odds",
        "markets": {
            "a_ko_odds_prob": ("am", +1050),
            "a_sub_odds_prob": ("am", +3000),
            "a_dec_odds_prob": ("am", +640),
            "b_ko_odds_prob": ("am", +425),
            "b_sub_odds_prob": ("am", +125),
            "b_dec_odds_prob": ("am", +225),
        },
    },
    {
        "event_date": "2025-07-12",
        "fighter_a": "Vitor Petrino",
        "fighter_b": "Austen Lane",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.com/sports/mixed-martial-arts/26420387/ufc-matches/10581356/vitor-petrino-v-austen-lane/34448888/",
        "markets": {
            "a_ko_odds_prob": ("dec", 1.40),
            "a_sub_odds_prob": ("dec", 8.00),
            "a_dec_odds_prob": ("dec", 9.50),
            "b_ko_odds_prob": ("dec", 11.00),
            "b_sub_odds_prob": ("dec", 26.00),
            "b_dec_odds_prob": ("dec", 18.00),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Adam Fugitt",
        "fighter_b": "Islam Dulatov",
        "source": "web_bodog_search",
        "source_url": "https://www.bodog.eu/sports/ufc-mma/ufc/ufc-318-holloway-poirier-3/adam-fugitt-islam-dulatov-202507191920",
        "markets": {
            "a_ko_odds_prob": ("am", +1000),
            "a_sub_odds_prob": ("am", +1600),
            "a_dec_odds_prob": ("am", +1100),
            "b_ko_odds_prob": ("am", -160),
            "b_sub_odds_prob": ("am", +275),
            "b_dec_odds_prob": ("am", +750),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Brunno Ferreira",
        "fighter_b": "Jackson McVey",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.com/sports/mixed-martial-arts/26420387/ufc-matches/10581356/brunno-ferreira-v-jackson-mcvey/34499981/",
        "markets": {
            "a_ko_odds_prob": ("dec", 1.73),
            "a_sub_odds_prob": ("dec", 3.30),
            "a_dec_odds_prob": ("dec", 18.00),
            "b_ko_odds_prob": ("dec", 13.00),
            "b_sub_odds_prob": ("dec", 13.00),
            "b_dec_odds_prob": ("dec", 17.00),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Carli Judice",
        "fighter_b": "Nicolle Caliari",
        "source": "web_bumbet_search",
        "source_url": "https://www.bumbet.com/en/sports/ufc-mma/ufc/ufc-318-holloway-poirier-3/carli-judice-nicolle-caliari-202507191815",
        "markets": {
            "a_ko_odds_prob": ("dec", 4.50),
            "a_sub_odds_prob": ("dec", 10.00),
            "a_dec_odds_prob": ("dec", 1.952),
            "b_ko_odds_prob": ("dec", 17.00),
            "b_sub_odds_prob": ("dec", 12.00),
            "b_dec_odds_prob": ("dec", 5.50),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Francisco Prado",
        "fighter_b": "Nikolay Veretennikov",
        "source": "web_pokerstars_search",
        "source_url": "https://www.pokerstars.es/en/sports/mixed-martial-arts/26420387/ufc-matches/10581356/francisco-prado-v-nikolay-veretennikov/34499976/",
        "markets": {
            "a_ko_odds_prob": ("dec", 4.50),
            "a_sub_odds_prob": ("dec", 12.00),
            "a_dec_odds_prob": ("dec", 3.10),
            "b_ko_odds_prob": ("dec", 7.00),
            "b_sub_odds_prob": ("dec", 15.00),
            "b_dec_odds_prob": ("dec", 3.25),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Jimmy Crute",
        "fighter_b": "Marcin Prachnio",
        "source": "web_everygame_search",
        "source_url": "https://sports.everygame.eu/en/Bets/Boxing-UFC/UFC-318/Marcin-Prachnio-v-Jimmy-Crute/2678965",
        "markets": {
            "a_ko_odds_prob": ("am", +375),
            "a_sub_odds_prob": ("am", +162),
            "a_dec_odds_prob": ("am", +300),
            "b_ko_odds_prob": ("am", +550),
            "b_sub_odds_prob": ("am", +2200),
            "b_dec_odds_prob": ("am", +500),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Kyler Phillips",
        "fighter_b": "Vinicius Oliveira",
        "source": "web_bovada_search",
        "source_url": "https://www.bovada.lv/sports/ufc-mma/ufc/ufc-318-holloway-poirier-3/kyler-phillips-vinicius-de-oliveira-202507192155?overlay=login",
        "markets": {
            "a_ko_odds_prob": ("am", +800),
            "a_sub_odds_prob": ("am", +1100),
            "a_dec_odds_prob": ("am", +250),
            "b_ko_odds_prob": ("am", +310),
            "b_sub_odds_prob": ("am", +1600),
            "b_dec_odds_prob": ("am", +165),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Marvin Vettori",
        "fighter_b": "Brendan Allen",
        "source": "web_covers",
        "source_url": "https://www.covers.com/ufc/318-predictions-prelims-picks-odds",
        "markets": {
            "a_ko_odds_prob": ("am", +1100),
            "a_sub_odds_prob": ("am", +1100),
            "a_dec_odds_prob": ("am", +275),
            "b_ko_odds_prob": ("am", +900),
            "b_sub_odds_prob": ("am", +500),
            "b_dec_odds_prob": ("am", +105),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Michael Johnson",
        "fighter_b": "Daniel Zellhuber",
        "source": "web_bodog_search",
        "source_url": "https://www.bodog.eu/sports/ufc-mma/ufc/ufc-318-holloway-poirier-3/michael-johnson-daniel-zellhuber-202507192215",
        "markets": {
            "a_ko_odds_prob": ("am", +1000),
            "a_sub_odds_prob": ("am", +4000),
            "a_dec_odds_prob": ("am", +750),
            "b_ko_odds_prob": ("am", +180),
            "b_sub_odds_prob": ("am", +425),
            "b_dec_odds_prob": ("am", +165),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Paulo Costa",
        "fighter_b": "Roman Kopylov",
        "source": "web_bodog_search",
        "source_url": "https://www.bodog.eu/sports/ufc-mma/ufc/ufc-318-holloway-poirier-3/paulo-costa-roman-kopylov-202507192345",
        "markets": {
            "a_ko_odds_prob": ("am", +550),
            "a_sub_odds_prob": ("am", +1600),
            "a_dec_odds_prob": ("am", +450),
            "b_ko_odds_prob": ("am", +300),
            "b_sub_odds_prob": ("am", +2800),
            "b_dec_odds_prob": ("am", +110),
        },
    },
    {
        "event_date": "2025-07-19",
        "fighter_a": "Ryan Spann",
        "fighter_b": "Lukasz Brzeski",
        "source": "web_betus_search",
        "source_url": "https://www.betus.com.pa/sportsbook/martial-arts/props/lukasz-brzeski-vs-ryan-spann/",
        "markets": {
            "a_ko_odds_prob": ("am", +160),
            "a_sub_odds_prob": ("am", +285),
            "a_dec_odds_prob": ("am", +600),
            "b_ko_odds_prob": ("am", +300),
            "b_sub_odds_prob": ("am", +1000),
            "b_dec_odds_prob": ("am", +800),
        },
    },
    {
        "event_date": "2025-08-22",
        "fighter_a": "Shi Ming",
        "fighter_b": "Bruna Brasil",
        "source": "web_clutchbuzz_search",
        "source_url": "https://www.clutchbuzz.bet/ufc/shi-ming-vs-bruna-brasil-prediction-road-to-ufc-shanghai-full-fight-preview-betting-picks/",
        "markets": {
            "a_sub_odds_prob": ("am", +400),
            "b_dec_odds_prob": ("am", +150),
        },
    },
]


def build_ml_row(item: dict) -> dict:
    a_dec = american_to_decimal(item["a_american"])
    b_dec = american_to_decimal(item["b_american"])
    a_imp = 1.0 / a_dec
    b_imp = 1.0 / b_dec
    total = a_imp + b_imp
    query_dt = datetime.strptime(item["query_date"], "%Y-%m-%d")
    event_dt = datetime.strptime(item["event_date"], "%Y-%m-%d")
    return {
        "event_date": item["event_date"],
        "fighter_a": item["fighter_a"],
        "fighter_b": item["fighter_b"],
        "query_date": item["query_date"],
        "offset_days": (event_dt - query_dt).days,
        "a_fair_prob": round(a_imp / total, 6),
        "b_fair_prob": round(b_imp / total, 6),
        "a_decimal_odds": round(a_dec, 4),
        "b_decimal_odds": round(b_dec, 4),
        "num_bookmakers": 1,
        "event_date_dt": item["event_date"],
        "fight_key": fight_key(item["event_date"], item["fighter_a"], item["fighter_b"]),
    }


def build_method_row(item: dict, captured_at: str) -> dict:
    row = {
        "event_date": item["event_date"],
        "fighter_a": item["fighter_a"],
        "fighter_b": item["fighter_b"],
        "event_title": "",
        "source": item["source"],
        "source_url": item["source_url"],
        "captured_at": captured_at,
        "a_ko_odds_prob": "",
        "a_sub_odds_prob": "",
        "a_dec_odds_prob": "",
        "b_ko_odds_prob": "",
        "b_sub_odds_prob": "",
        "b_dec_odds_prob": "",
    }
    for field, (kind, value) in item["markets"].items():
        row[field] = round(odds_value_to_prob(kind, value), 6)
    return row


def rebuild_missing_moneyline(existing_ml: pd.DataFrame) -> list[dict]:
    keys = set(existing_ml["fight_key"].dropna().astype(str))
    missing = list(csv.DictReader(open(MISSING_ML_PATH, encoding="utf-8")))
    remaining = []
    for row in missing:
        if fight_key(row["event_date"], row["fighter_a"], row["fighter_b"]) not in keys:
            remaining.append(row)
    return remaining


def rebuild_missing_method(existing_method: pd.DataFrame) -> list[dict]:
    keys = set()
    for _, row in existing_method.iterrows():
        k1 = (str(row["event_date"])[:10], norm(row["fighter_a"]), norm(row["fighter_b"]))
        k2 = (str(row["event_date"])[:10], norm(row["fighter_b"]), norm(row["fighter_a"]))
        keys.add(k1)
        keys.add(k2)

    missing = list(csv.DictReader(open(MISSING_METHOD_PATH, encoding="utf-8")))
    remaining = []
    for row in missing:
        key = (row["event_date"], norm(row["fighter_a"]), norm(row["fighter_b"]))
        if key not in keys:
            remaining.append(row)
    return remaining


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    existing_ml = pd.read_csv(ML_PATH, low_memory=False)
    existing_ml_keys = set(existing_ml["fight_key"].dropna().astype(str))
    ml_new = []
    for item in RECOVERED_MONEYLINES:
        row = build_ml_row(item)
        if row["fight_key"] not in existing_ml_keys:
            ml_new.append(row)
            existing_ml_keys.add(row["fight_key"])

    if ml_new:
        with open(ML_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(existing_ml.columns))
            for row in ml_new:
                writer.writerow({field: row.get(field, "") for field in existing_ml.columns})

    existing_ml = pd.read_csv(ML_PATH, low_memory=False)
    remaining_ml = rebuild_missing_moneyline(existing_ml)
    ml_fields = ["event_date", "fighter_a", "fighter_b"]
    write_csv(MISSING_ML_PATH, remaining_ml, ml_fields)
    write_csv(STILL_MISSING_ML_PATH, remaining_ml, ml_fields)

    existing_method = pd.read_csv(METHOD_PATH, low_memory=False)
    existing_method_keys = set()
    for _, row in existing_method.iterrows():
        existing_method_keys.add((str(row["event_date"])[:10], norm(row["fighter_a"]), norm(row["fighter_b"])))
        existing_method_keys.add((str(row["event_date"])[:10], norm(row["fighter_b"]), norm(row["fighter_a"])))

    method_new = []
    for item in RECOVERED_METHOD_ROWS:
        key = (item["event_date"], norm(item["fighter_a"]), norm(item["fighter_b"]))
        if key not in existing_method_keys:
            row = build_method_row(item, captured_at)
            method_new.append(row)
            existing_method_keys.add(key)
            existing_method_keys.add((item["event_date"], norm(item["fighter_b"]), norm(item["fighter_a"])))

    if method_new:
        with open(METHOD_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(existing_method.columns))
            for row in method_new:
                writer.writerow({field: row.get(field, "") for field in existing_method.columns})

    existing_method = pd.read_csv(METHOD_PATH, low_memory=False)
    remaining_method = rebuild_missing_method(existing_method)
    write_csv(MISSING_METHOD_PATH, remaining_method, ["event_date", "fighter_a", "fighter_b"])

    print(f"Moneyline appended: {len(ml_new)}")
    print(f"Moneyline remaining: {len(remaining_ml)}")
    print(f"Method rows appended: {len(method_new)}")
    print(f"Method rows remaining: {len(remaining_method)}")


if __name__ == "__main__":
    main()
