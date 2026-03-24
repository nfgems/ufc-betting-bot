import numpy as np
import pandas as pd
import pytest

from src.config import ELO_INITIAL
from src.features.tennis_features import (
    build_tennis_features,
    build_live_tennis_features,
    build_tournament_context,
    enrich_matchups_with_context,
    get_stage1_tennis_feature_columns,
    get_stage2_tennis_feature_columns,
    get_tennis_feature_columns,
)


def test_build_tennis_features_is_leak_free_across_matches():
    matches = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2024-01-01"),
                "tourney_name": "Test Open",
                "match_num": 1,
                "tourney_level": "A",
                "surface": "Clay",
                "best_of": 3,
                "player_a": "Rafael Nadal",
                "player_b": "Roger Federer",
                "winner": "Rafael Nadal",
                "target": 1,
                "player_a_rank": 2,
                "player_b_rank": 5,
                "player_a_rank_points": 6000,
                "player_b_rank_points": 4200,
                "player_a_age": 37.0,
                "player_b_age": 42.0,
                "player_a_hand": "L",
                "player_b_hand": "R",
            },
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2024-02-01"),
                "tourney_name": "Test Open",
                "match_num": 2,
                "tourney_level": "A",
                "surface": "Clay",
                "best_of": 3,
                "player_a": "Rafael Nadal",
                "player_b": "Novak Djokovic",
                "winner": "Novak Djokovic",
                "target": 0,
                "player_a_rank": 3,
                "player_b_rank": 1,
                "player_a_rank_points": 5800,
                "player_b_rank_points": 7100,
                "player_a_age": 37.0,
                "player_b_age": 36.0,
                "player_a_hand": "L",
                "player_b_hand": "R",
            },
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2024-03-01"),
                "tourney_name": "Test Open",
                "match_num": 3,
                "tourney_level": "A",
                "surface": "Clay",
                "best_of": 3,
                "player_a": "Rafael Nadal",
                "player_b": "Roger Federer",
                "winner": "Roger Federer",
                "target": 0,
                "player_a_rank": 4,
                "player_b_rank": 7,
                "player_a_rank_points": 5600,
                "player_b_rank_points": 3500,
                "player_a_age": 37.0,
                "player_b_age": 42.0,
                "player_a_hand": "L",
                "player_b_hand": "R",
            },
        ]
    )

    features = build_tennis_features(matches)

    assert features.loc[0, "a_num_matches"] == 0
    assert features.loc[0, "b_num_matches"] == 0
    assert np.isnan(features.loc[0, "a_recent_form"])

    assert features.loc[1, "a_num_matches"] == 1
    assert features.loc[1, "b_num_matches"] == 0
    assert features.loc[1, "a_surface_win_rate"] == 1.0
    assert features.loc[1, "a_days_since_last_match"] == 31.0
    assert features.loc[1, "a_rank_points"] == 5800.0
    assert features.loc[1, "log_rank_points_diff"] == pytest.approx(np.log1p(5800) - np.log1p(7100))

    assert features.loc[2, "a_num_matches"] == 2
    assert features.loc[2, "b_num_matches"] == 1
    assert features.loc[2, "a_h2h_wins"] == 1.0
    assert features.loc[2, "b_h2h_wins"] == 0.0


def test_build_tournament_context_keeps_atp_and_wta_separate_for_same_seed():
    history = pd.DataFrame(
        [
            {
                "tour": "atp",
                "tourney_name": "Australian Open",
                "tournament_seed": "australian open",
                "surface": "Hard",
                "best_of": 5,
                "tourney_level": "G",
            },
            {
                "tour": "wta",
                "tourney_name": "Australian Open",
                "tournament_seed": "australian open",
                "surface": "Hard",
                "best_of": 3,
                "tourney_level": "G",
            },
        ]
    )

    context = build_tournament_context(history)

    assert len(context) == 2
    atp_best_of = context.loc[context["tour"] == "atp", "best_of"].iloc[0]
    wta_best_of = context.loc[context["tour"] == "wta", "best_of"].iloc[0]
    assert atp_best_of == 5
    assert wta_best_of == 3


def test_build_tennis_features_does_not_leak_same_name_history_across_tours():
    matches = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2024-01-01"),
                "tourney_name": "ATP Test Open",
                "match_num": 1,
                "tourney_level": "A",
                "surface": "Hard",
                "best_of": 3,
                "player_a": "Alex Kim",
                "player_b": "Tour Opponent",
                "winner": "Alex Kim",
                "target": 1,
                "player_a_rank": 40,
                "player_b_rank": 80,
                "player_a_age": 24.0,
                "player_b_age": 27.0,
                "player_a_hand": "R",
                "player_b_hand": "R",
            },
            {
                "tour": "wta",
                "event_date": pd.Timestamp("2024-01-02"),
                "tourney_name": "WTA Test Open",
                "match_num": 1,
                "tourney_level": "A",
                "surface": "Hard",
                "best_of": 3,
                "player_a": "Alex Kim",
                "player_b": "Second Opponent",
                "winner": "Second Opponent",
                "target": 0,
                "player_a_rank": 22,
                "player_b_rank": 18,
                "player_a_age": 21.0,
                "player_b_age": 25.0,
                "player_a_hand": "L",
                "player_b_hand": "R",
            },
        ]
    )

    features = build_tennis_features(matches)
    wta_row = features.loc[features["tour"] == "wta"].iloc[0]

    assert wta_row["a_num_matches"] == 0
    assert wta_row["a_elo"] == ELO_INITIAL
    assert np.isnan(wta_row["a_recent_form"])


def test_enrich_matchups_with_context_does_not_cross_contaminate_tours():
    history = pd.DataFrame(
        [
            {
                "tour": "atp",
                "tourney_name": "Australian Open",
                "tournament_seed": "australian open",
                "surface": "Hard",
                "best_of": 5,
                "tourney_level": "G",
            },
            {
                "tour": "wta",
                "tourney_name": "Australian Open",
                "tournament_seed": "australian open",
                "surface": "Hard",
                "best_of": 3,
                "tourney_level": "G",
            },
        ]
    )
    matchups = pd.DataFrame(
        [
            {
                "tour": "atp",
                "fighter_a": "Novak Djokovic",
                "fighter_b": "Jannik Sinner",
                "tournament_seed": "australian open",
            },
            {
                "tour": "wta",
                "fighter_a": "Iga Świątek",
                "fighter_b": "Aryna Sabalenka",
                "tournament_seed": "australian open",
            },
        ]
    )

    enriched = enrich_matchups_with_context(matchups, history)

    assert enriched.loc[enriched["tour"] == "atp", "best_of"].iloc[0] == 5
    assert enriched.loc[enriched["tour"] == "wta", "best_of"].iloc[0] == 3


def test_enrich_matchups_with_context_rejects_distinct_tournaments_in_same_city():
    history = pd.DataFrame(
        [
            {
                "tour": "atp",
                "tourney_name": "Paris Open",
                "tournament_seed": "paris open",
                "surface": "Clay",
                "best_of": 3,
                "tourney_level": "A",
            }
        ]
    )
    matchups = pd.DataFrame(
        [
            {
                "tour": "atp",
                "fighter_a": "Novak Djokovic",
                "fighter_b": "Jannik Sinner",
                "tournament_seed": "paris masters",
            }
        ]
    )

    enriched = enrich_matchups_with_context(matchups, history)

    assert enriched.loc[0, "surface"] == "unknown"


def test_enrich_matchups_with_context_uses_atp_grand_slam_fallback_when_lookup_misses():
    history = pd.DataFrame(
        [
            {
                "tour": "atp",
                "tourney_name": "Miami Open",
                "tournament_seed": "miami open",
                "surface": "Hard",
                "best_of": 3,
                "tourney_level": "M",
            }
        ]
    )
    matchups = pd.DataFrame(
        [
            {
                "tour": "atp",
                "fighter_a": "Novak Djokovic",
                "fighter_b": "Jannik Sinner",
                "sport_key": "tennis_atp_wimbledon",
                "tournament_name": "Wimbledon Men Singles",
            }
        ]
    )

    enriched = enrich_matchups_with_context(matchups, history)

    assert enriched.loc[0, "best_of"] == 5
    assert enriched.loc[0, "tourney_level"] == "G"


def test_enrich_matchups_with_context_handles_empty_history_defaults():
    matchups = pd.DataFrame(
        [
            {
                "tour": "wta",
                "fighter_a": "Iga Świątek",
                "fighter_b": "Coco Gauff",
                "sport_key": "tennis_wta_indian_wells",
            }
        ]
    )

    enriched = enrich_matchups_with_context(matchups, pd.DataFrame())

    assert enriched.loc[0, "best_of"] == 3
    assert enriched.loc[0, "surface"] == "unknown"


def test_tennis_feature_column_helpers_make_stage1_contract_explicit():
    features_df = pd.DataFrame(
        columns=[
            "a_elo",
            "diff_elo",
            "diff_surface_elo",
            "best_of_5",
            "log_rank_diff",
            "log_rank_points_diff",
            "diff_age",
            "same_hand",
        ]
    )

    assert get_stage1_tennis_feature_columns(features_df) == [
        "diff_elo",
        "diff_surface_elo",
        "best_of_5",
    ]
    assert get_stage2_tennis_feature_columns(features_df) == [
        "diff_elo",
        "diff_surface_elo",
        "best_of_5",
        "log_rank_diff",
        "log_rank_points_diff",
        "diff_age",
    ]
    assert get_tennis_feature_columns(features_df) == [
        "a_elo",
        "diff_elo",
        "diff_surface_elo",
        "log_rank_diff",
        "log_rank_points_diff",
        "best_of_5",
        "diff_age",
        "same_hand",
    ]


def test_build_live_tennis_features_carries_forward_rank_points_state():
    history = pd.DataFrame(
        [
            {
                "tour": "wta",
                "event_date": pd.Timestamp("2026-03-10"),
                "tourney_name": "History Event",
                "match_num": 1,
                "surface": "Hard",
                "best_of": 3,
                "tourney_level": "A",
                "player_a": "Player One",
                "player_b": "Player Two",
                "winner": "Player One",
                "target": 1,
                "player_a_rank": 12,
                "player_b_rank": 30,
                "player_a_rank_points": 2850,
                "player_b_rank_points": 1625,
                "player_a_age": 24.0,
                "player_b_age": 27.0,
                "player_a_hand": "R",
                "player_b_hand": "L",
            }
        ]
    )
    live_matchups = pd.DataFrame(
        [
            {
                "tour": "wta",
                "fighter_a": "Player One",
                "fighter_b": "Player Two",
                "event_title": "Player One vs Player Two",
                "commence_time": pd.Timestamp("2026-03-22"),
            }
        ]
    )

    features = build_live_tennis_features(live_matchups, history)

    assert features.loc[0, "a_rank_points"] == 2850.0
    assert features.loc[0, "b_rank_points"] == 1625.0
    assert features.loc[0, "log_rank_points_diff"] == pytest.approx(np.log1p(2850) - np.log1p(1625))


def test_build_live_tennis_features_handles_timezone_aware_commence_time():
    history = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2026-03-20"),
                "tourney_name": "History Event",
                "match_num": 1,
                "surface": "Hard",
                "best_of": 3,
                "tourney_level": "A",
                "player_a": "Player One",
                "player_b": "Player Two",
                "winner": "Player One",
                "target": 1,
                "player_a_rank": 12,
                "player_b_rank": 30,
                "player_a_rank_points": 2850,
                "player_b_rank_points": 1625,
                "player_a_age": 24.0,
                "player_b_age": 27.0,
                "player_a_hand": "R",
                "player_b_hand": "L",
            }
        ]
    )
    live_matchups = pd.DataFrame(
        [
            {
                "tour": "atp",
                "fighter_a": "Player One",
                "fighter_b": "Player Two",
                "event_title": "Player One vs Player Two",
                "commence_time": "2026-03-22T15:30:00Z",
            }
        ]
    )

    features = build_live_tennis_features(live_matchups, history)

    assert features.loc[0, "a_days_since_last_match"] == pytest.approx(2.0)
    assert features.loc[0, "b_days_since_last_match"] == pytest.approx(2.0)
    assert features.loc[0, "a_matches_last_7d"] == pytest.approx(1.0)
    assert features.loc[0, "b_matches_last_7d"] == pytest.approx(1.0)
