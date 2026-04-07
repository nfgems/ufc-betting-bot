import pytest

from src.data.odds_client import OddsClient


def _event(*, home_team: str, away_team: str, home_price: float, away_price: float) -> dict:
    return {
        "id": "evt-1",
        "commence_time": "2026-04-12T21:00:00Z",
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": [
            {
                "title": "Book A",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home_team, "price": home_price},
                            {"name": away_team, "price": away_price},
                        ],
                    }
                ],
            }
        ],
    }


def test_odds_to_dataframe_normalizes_fighter_order_and_keeps_matching_odds():
    client = OddsClient(api_key="test")

    first = client.odds_to_dataframe(
        [
            _event(
                home_team="Jiri Prochazka",
                away_team="Carlos Ulberg",
                home_price=2.30,
                away_price=1.62,
            )
        ]
    )
    second = client.odds_to_dataframe(
        [
            _event(
                home_team="Carlos Ulberg",
                away_team="Jiri Prochazka",
                home_price=1.62,
                away_price=2.30,
            )
        ]
    )

    assert len(first) == 1
    assert len(second) == 1

    first_row = first.iloc[0].to_dict()
    second_row = second.iloc[0].to_dict()

    assert first_row["fighter_a"] == "Carlos Ulberg"
    assert first_row["fighter_b"] == "Jiri Prochazka"
    assert second_row["fighter_a"] == "Carlos Ulberg"
    assert second_row["fighter_b"] == "Jiri Prochazka"

    assert first_row["a_odds"] == pytest.approx(1.62)
    assert first_row["b_odds"] == pytest.approx(2.30)
    assert second_row["a_odds"] == pytest.approx(1.62)
    assert second_row["b_odds"] == pytest.approx(2.30)
    assert first_row["a_implied_prob"] == pytest.approx(second_row["a_implied_prob"])
    assert first_row["b_implied_prob"] == pytest.approx(second_row["b_implied_prob"])


def test_get_consensus_odds_is_stable_when_api_home_away_flips():
    client = OddsClient(api_key="test")

    odds_df = client.odds_to_dataframe(
        [
            _event(
                home_team="Jiri Prochazka",
                away_team="Carlos Ulberg",
                home_price=2.30,
                away_price=1.62,
            ),
            _event(
                home_team="Carlos Ulberg",
                away_team="Jiri Prochazka",
                home_price=1.62,
                away_price=2.30,
            ),
        ]
    )

    consensus = client.get_consensus_odds(odds_df)

    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["fighter_a"] == "Carlos Ulberg"
    assert row["fighter_b"] == "Jiri Prochazka"
    assert row["a_odds_avg"] == pytest.approx(1.62)
    assert row["b_odds_avg"] == pytest.approx(2.30)
    assert row["num_bookmakers"] == 2
