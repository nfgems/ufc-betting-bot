import pandas as pd

from src.data.tennis_data import (
    derive_tournament_seed,
    normalize_tennis_matches,
    tournament_seeds_compatible,
)


def test_derive_tournament_seed_normalizes_common_tournament_aliases():
    assert derive_tournament_seed("Roland Garros") == "french open"
    assert derive_tournament_seed("United States Open") == "us open"
    assert derive_tournament_seed("U.S. Open Tennis Championships") == "us open"


def test_derive_tournament_seed_preserves_numeric_suffixes():
    assert derive_tournament_seed("Adelaide 1") == "adelaide 1"
    assert derive_tournament_seed("Adelaide 2") == "adelaide 2"
    assert derive_tournament_seed("Miami Open 2024") == "miami open"


def test_tournament_seeds_compatible_rejects_location_only_false_positives():
    assert tournament_seeds_compatible("Paris Masters", "paris open") is False
    assert tournament_seeds_compatible("Adelaide 1", "Adelaide 2") is False


def test_normalize_tennis_matches_excludes_retirements_and_orients_players():
    raw = pd.DataFrame(
        [
            {
                "tour": "atp",
                "tourney_id": "2024-001",
                "tourney_name": "Test Open",
                "tourney_level": "A",
                "surface": "Hard",
                "draw_size": 32,
                "tourney_date": 20240101,
                "match_num": 1,
                "round": "R32",
                "best_of": 3,
                "minutes": 95,
                "score": "6-4 6-4",
                "winner_name": "Rafael Nadal",
                "loser_name": "Carlos Alcaraz",
                "winner_id": 1,
                "loser_id": 2,
                "winner_rank": 2,
                "loser_rank": 1,
                "winner_rank_points": 8000,
                "loser_rank_points": 9000,
                "winner_age": 37.0,
                "loser_age": 20.0,
                "winner_hand": "L",
                "loser_hand": "R",
                "winner_ht": 185,
                "loser_ht": 183,
                "winner_seed": 2,
                "loser_seed": 1,
                "winner_entry": None,
                "loser_entry": None,
            },
            {
                "tour": "atp",
                "tourney_id": "2024-001",
                "tourney_name": "Test Open",
                "tourney_level": "A",
                "surface": "Hard",
                "draw_size": 32,
                "tourney_date": 20240102,
                "match_num": 2,
                "round": "R32",
                "best_of": 3,
                "minutes": 45,
                "score": "6-1 2-0 RET",
                "winner_name": "Novak Djokovic",
                "loser_name": "Andy Murray",
                "winner_id": 3,
                "loser_id": 4,
                "winner_rank": 1,
                "loser_rank": 40,
                "winner_rank_points": 10000,
                "loser_rank_points": 1200,
                "winner_age": 36.0,
                "loser_age": 36.0,
                "winner_hand": "R",
                "loser_hand": "R",
                "winner_ht": 188,
                "loser_ht": 191,
                "winner_seed": 1,
                "loser_seed": None,
                "winner_entry": None,
                "loser_entry": None,
            },
        ]
    )

    normalized = normalize_tennis_matches(raw)

    assert len(normalized) == 1
    assert normalized.loc[0, "winner"] == "Rafael Nadal"
    assert normalized.loc[0, "player_a"] == "Carlos Alcaraz"
    assert normalized.loc[0, "player_b"] == "Rafael Nadal"
    assert normalized.loc[0, "target"] == 0
