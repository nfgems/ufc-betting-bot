import pandas as pd

from src.polymarket.tennis_markets import (
    _event_matches_seed,
    extract_players_from_title,
    match_tennis_markets,
    normalize_tennis_market_names,
)


def test_extract_players_from_title_parses_tennis_match_title():
    assert extract_players_from_title("Dubai Tennis Championships: Quentin Halys vs Jack Draper") == (
        "Quentin Halys",
        "Jack Draper",
    )


def test_normalize_tennis_market_names_maps_short_outcomes_to_full_names():
    mapped = normalize_tennis_market_names(
        players=("Novak Djokovic", "Carlos Alcaraz"),
        outcomes=["Djokovic", "Alcaraz"],
        event_title="ATP Finals: Novak Djokovic vs Carlos Alcaraz",
    )

    assert mapped == ("Novak Djokovic", "Carlos Alcaraz")


def test_normalize_tennis_market_names_returns_none_for_ambiguous_outcomes():
    mapped = normalize_tennis_market_names(
        players=("Serena Williams", "Venus Williams"),
        outcomes=["Williams", "Williams"],
        event_title="Exhibition: Serena Williams vs Venus Williams",
    )

    assert mapped is None


def test_match_tennis_markets_flips_token_orientation_to_match_fighter_a():
    matchups = pd.DataFrame(
        [
            {
                "fighter_a": "Carlos Alcaraz",
                "fighter_b": "Novak Djokovic",
                "tour": "atp",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "123",
                "condition_id": "cond-1",
                "fighter_a": "Novak Djokovic",
                "fighter_b": "Carlos Alcaraz",
                "token_id_yes": "token-novak",
                "token_id_no": "token-carlos",
                "price_yes": 0.60,
                "price_no": 0.40,
                "event_title": "ATP Finals: Novak Djokovic vs Carlos Alcaraz",
                "event_date": "2026-03-12T12:00:00Z",
                "tour": "atp",
                "liquidity": 1500,
                "tick_size": "0.01",
            }
        ]
    )

    matched = match_tennis_markets(matchups, markets)

    assert len(matched) == 1
    assert matched.loc[0, "token_id_yes"] == "token-carlos"
    assert matched.loc[0, "token_id_no"] == "token-novak"
    assert matched.loc[0, "price_yes"] == 0.40
    assert matched.loc[0, "price_no"] == 0.60


def test_match_tennis_markets_matches_surname_only_market_names_to_full_bookmaker_names():
    matchups = pd.DataFrame(
        [
            {
                "fighter_a": "Francisco Cerundolo",
                "fighter_b": "Grigor Dimitrov",
                "tour": "atp",
                "tournament_seed": "miami open",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "miami-1",
                "condition_id": "cond-miami-1",
                "fighter_a": "Cerundolo",
                "fighter_b": "Dimitrov",
                "token_id_yes": "token-cerundolo",
                "token_id_no": "token-dimitrov",
                "price_yes": 0.47,
                "price_no": 0.53,
                "question": "Who will win?",
                "event_title": "Miami Open: Cerundolo vs Dimitrov",
                "tour": "atp",
                "tournament_seed": "miami open",
                "liquidity": 2200,
                "tick_size": "0.01",
            }
        ]
    )

    matched = match_tennis_markets(matchups, markets)

    assert len(matched) == 1
    assert matched.loc[0, "token_id_yes"] == "token-cerundolo"
    assert matched.loc[0, "token_id_no"] == "token-dimitrov"


def test_match_tennis_markets_matches_abbreviated_market_names_to_full_bookmaker_names():
    matchups = pd.DataFrame(
        [
            {
                "fighter_a": "Francisco Cerundolo",
                "fighter_b": "Grigor Dimitrov",
                "tour": "atp",
                "tournament_seed": "miami open",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "miami-2",
                "condition_id": "cond-miami-2",
                "fighter_a": "F. Cerundolo",
                "fighter_b": "G. Dimitrov",
                "token_id_yes": "token-cerundolo",
                "token_id_no": "token-dimitrov",
                "price_yes": 0.49,
                "price_no": 0.51,
                "question": "Who will win?",
                "event_title": "Miami Open: F. Cerundolo vs G. Dimitrov",
                "tour": "atp",
                "tournament_seed": "miami open",
                "liquidity": 2100,
                "tick_size": "0.01",
            }
        ]
    )

    matched = match_tennis_markets(matchups, markets)

    assert len(matched) == 1
    assert matched.loc[0, "market_id"] == "miami-2"


def test_match_tennis_markets_normalizes_non_ascii_wta_names():
    matchups = pd.DataFrame(
        [
            {
                "fighter_a": "Iga Świątek",
                "fighter_b": "Aryna Sabalenka",
                "tour": "wta",
                "tournament_seed": "indian wells",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "iw-1",
                "condition_id": "cond-iw-1",
                "fighter_a": "Swiatek",
                "fighter_b": "Sabalenka",
                "token_id_yes": "token-swiatek",
                "token_id_no": "token-sabalenka",
                "price_yes": 0.61,
                "price_no": 0.39,
                "question": "Who will win?",
                "event_title": "Indian Wells: Swiatek vs Sabalenka",
                "tour": "wta",
                "tournament_seed": "indian wells",
                "liquidity": 1800,
                "tick_size": "0.01",
            }
        ]
    )

    matched = match_tennis_markets(matchups, markets)

    assert len(matched) == 1
    assert matched.loc[0, "token_id_yes"] == "token-swiatek"


def test_event_matches_seed_rejects_generic_token_false_positive():
    event = {
        "title": "US Open: Djokovic vs Alcaraz",
        "slug": "us-open-djokovic-vs-alcaraz",
        "category": "Tennis",
        "subcategory": "Men's Singles",
    }

    assert _event_matches_seed(event, "us open") is True
    assert _event_matches_seed(event, "miami open") is False
    assert _event_matches_seed(event, "australian open") is False


def test_event_matches_seed_rejects_distinct_tournaments_in_same_city():
    event = {
        "title": "Paris Open: Djokovic vs Alcaraz",
        "slug": "paris-open-djokovic-vs-alcaraz",
        "category": "Tennis",
        "subcategory": "Men's Singles",
    }

    assert _event_matches_seed(event, "paris open") is True
    assert _event_matches_seed(event, "paris masters") is False
