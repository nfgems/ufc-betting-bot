from src.data.tennis_odds import filter_active_tennis_sports, parse_tour_from_sport_key


def test_parse_tour_from_sport_key_handles_atp_and_wta():
    assert parse_tour_from_sport_key("tennis_atp_miami_open") == "atp"
    assert parse_tour_from_sport_key("tennis_wta_indian_wells") == "wta"
    assert parse_tour_from_sport_key("tennis_generic") == ""


def test_filter_active_tennis_sports_keeps_only_active_tournament_keys():
    sports = [
        {"key": "mma_mixed_martial_arts", "active": True, "title": "MMA", "description": "MMA"},
        {
            "key": "tennis_atp_miami_open",
            "active": True,
            "title": "ATP Miami",
            "description": "Miami Open Men Singles",
        },
        {
            "key": "tennis_wta_indian_wells",
            "active": True,
            "title": "WTA Indian Wells",
            "description": "Indian Wells Women Singles",
        },
        {
            "key": "tennis_atp_paris_masters",
            "active": False,
            "title": "ATP Paris",
            "description": "Paris Masters Men Singles",
        },
        {
            "key": "tennis",
            "active": True,
            "title": "Generic Tennis",
            "description": "Generic Tennis",
        },
    ]

    filtered = filter_active_tennis_sports(sports)

    assert [item["sport_key"] for item in filtered] == [
        "tennis_atp_miami_open",
        "tennis_wta_indian_wells",
    ]
    assert filtered[0]["tour"] == "atp"
    assert filtered[0]["tournament_seed"] == "miami open"
    assert filtered[1]["tour"] == "wta"
    assert filtered[1]["tournament_seed"] == "indian wells"
