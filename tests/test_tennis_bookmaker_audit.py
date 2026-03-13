import sys

import pandas as pd

import src.bot as bot
from src.data.tennis_bookmaker_audit import (
    build_join_by_year,
    evaluate_audit_summary,
    match_ground_truth_to_bookmaker_events,
    prepare_tennis_ground_truth,
)


def _ground_truth_row(
    *,
    event_date: str,
    tour: str,
    tourney_name: str,
    tournament_seed: str,
    player_a: str,
    player_b: str,
    match_num: int = 1,
) -> dict[str, object]:
    return {
        "event_date": event_date,
        "tour": tour,
        "tourney_name": tourney_name,
        "tournament_seed": tournament_seed,
        "player_a": player_a,
        "player_b": player_b,
        "match_num": match_num,
    }


def test_match_ground_truth_to_bookmaker_events_joins_on_normalized_names_and_tournament_seed():
    ground_truth = prepare_tennis_ground_truth(
        pd.DataFrame(
            [
                _ground_truth_row(
                    event_date="2024-05-20",
                    tour="wta",
                    tourney_name="Internationaux de Strasbourg",
                    tournament_seed="strasbourg",
                    player_a="Beatriz Haddad Maia",
                    player_b="Daria Kasatkina",
                )
            ]
        )
    )
    bookmaker_events = pd.DataFrame(
        [
            {
                "source": "odds_api",
                "source_event_id": "evt-1",
                "tour": "wta",
                "tournament_name": "WTA Strasbourg Women Singles",
                "tournament_seed": "strasbourg",
                "player_a": "B. Haddad Maia",
                "player_b": "Daria Kasatkina",
                "scheduled_match_start": "2024-05-22T10:00:00Z",
                "snapshot_timestamp": "2024-05-22T08:00:00Z",
                "bookmaker_last_update": "2024-05-22T08:30:00Z",
                "bookmaker_count": 4,
                "source_context": "tennis_wta_strasbourg",
            }
        ]
    )

    joined, unmatched = match_ground_truth_to_bookmaker_events(ground_truth, bookmaker_events)

    assert len(joined) == 1
    assert joined.loc[0, "source_event_id"] == "evt-1"
    assert joined.loc[0, "candidate_count"] == 1
    assert unmatched.empty


def test_match_ground_truth_to_bookmaker_events_falls_back_to_tournament_name_lookup():
    ground_truth = prepare_tennis_ground_truth(
        pd.DataFrame(
            [
                _ground_truth_row(
                    event_date="2024-04-22",
                    tour="atp",
                    tourney_name="Mutua Madrid Open",
                    tournament_seed="madrid",
                    player_a="Carlos Alcaraz",
                    player_b="Jannik Sinner",
                )
            ]
        )
    )
    bookmaker_events = pd.DataFrame(
        [
            {
                "source": "odds_api",
                "source_event_id": "evt-name-fallback",
                "tour": "atp",
                "tournament_name": "Mutua Madrid Open",
                "tournament_seed": "madrid-open",
                "player_a": "C. Alcaraz",
                "player_b": "J. Sinner",
                "scheduled_match_start": "2024-04-24T10:00:00Z",
                "snapshot_timestamp": "2024-04-24T07:45:00Z",
                "bookmaker_last_update": "2024-04-24T08:10:00Z",
                "bookmaker_count": 6,
                "source_context": "tennis_atp_madrid",
            }
        ]
    )

    joined, unmatched = match_ground_truth_to_bookmaker_events(ground_truth, bookmaker_events)

    assert len(joined) == 1
    assert joined.loc[0, "source_event_id"] == "evt-name-fallback"
    assert unmatched.empty


def test_build_join_by_year_tracks_match_counts_by_year_and_tour():
    ground_truth = prepare_tennis_ground_truth(
        pd.DataFrame(
            [
                _ground_truth_row(
                    event_date="2022-01-03",
                    tour="atp",
                    tourney_name="Adelaide",
                    tournament_seed="adelaide",
                    player_a="Player A1",
                    player_b="Player A2",
                ),
                _ground_truth_row(
                    event_date="2022-01-03",
                    tour="wta",
                    tourney_name="Adelaide",
                    tournament_seed="adelaide",
                    player_a="Player B1",
                    player_b="Player B2",
                ),
                _ground_truth_row(
                    event_date="2023-01-02",
                    tour="atp",
                    tourney_name="Auckland",
                    tournament_seed="auckland",
                    player_a="Player C1",
                    player_b="Player C2",
                ),
                _ground_truth_row(
                    event_date="2023-01-02",
                    tour="wta",
                    tourney_name="Auckland",
                    tournament_seed="auckland",
                    player_a="Player D1",
                    player_b="Player D2",
                ),
            ]
        )
    )
    joined = pd.DataFrame(
        [
            {"audit_row_id": int(ground_truth.loc[0, "audit_row_id"]), "event_year": 2022, "tour": "atp", "tournament_seed": "adelaide"},
            {"audit_row_id": int(ground_truth.loc[3, "audit_row_id"]), "event_year": 2023, "tour": "wta", "tournament_seed": "auckland"},
        ]
    )

    by_year = build_join_by_year(ground_truth, joined)

    atp_2022 = by_year[(by_year["event_year"] == 2022) & (by_year["tour"] == "atp")].iloc[0]
    wta_2023 = by_year[(by_year["event_year"] == 2023) & (by_year["tour"] == "wta")].iloc[0]

    assert atp_2022["matched_rows"] == 1
    assert atp_2022["unmatched_rows"] == 0
    assert wta_2023["matched_rows"] == 1
    assert wta_2023["unmatched_rows"] == 0


def test_evaluate_audit_summary_rejects_non_pre_match_timestamps():
    ground_truth = prepare_tennis_ground_truth(
        pd.DataFrame(
            [
                _ground_truth_row(
                    event_date=f"{year}-01-02",
                    tour=tour,
                    tourney_name=f"Tournament {year} {tour}",
                    tournament_seed=f"tournament {year} {tour}",
                    player_a=f"{tour} A {year}",
                    player_b=f"{tour} B {year}",
                )
                for year in range(2022, 2027)
                for tour in ["atp", "wta"]
            ]
        )
    )
    join_by_year = pd.DataFrame(
        [
            {
                "event_year": year,
                "tour": tour,
                "ground_truth_rows": 1,
                "ground_truth_tournaments": 1,
                "matched_rows": 1,
                "matched_tournaments": 1,
                "unmatched_rows": 0,
                "matched_rate": 1.0,
            }
            for year in range(2022, 2027)
            for tour in ["atp", "wta"]
        ]
    )
    joined = pd.DataFrame(
        [
            {
                "candidate_count": 1,
                "event_year": 2024,
                "tour": "atp",
                "player_a": "Player A",
                "player_b": "Player B",
                "source_event_id": "evt-1",
                "scheduled_match_start": "2024-05-22T10:00:00Z",
                "snapshot_timestamp": "2024-05-22T10:30:00Z",
                "bookmaker_last_update": "2024-05-22T10:40:00Z",
            }
        ]
    )

    summary = evaluate_audit_summary(
        source="betsapi",
        ground_truth_df=ground_truth,
        joined_df=joined,
        join_by_year=join_by_year,
        hard_fail=False,
        reason="",
        coverage_meta={},
        coverage_df=None,
    )

    assert summary["verdict"] == "no_go"
    assert summary["timestamp_metrics"]["timestamp_all_pass"] is False
    assert any("snapshot_timestamp < scheduled_match_start" in reason for reason in summary["reasons"])


def test_tennis_bookmaker_audit_cli_smoke(monkeypatch):
    captured = {}

    def _capture(args):
        captured["source"] = args.source
        captured["start_date"] = args.start_date
        captured["end_date"] = args.end_date

    monkeypatch.setattr(bot, "cmd_tennis_bookmaker_audit", _capture)
    monkeypatch.setattr(sys, "argv", ["bot", "tennis-bookmaker-audit"])

    bot.main()

    assert captured == {
        "source": "auto",
        "start_date": "2022-01-01",
        "end_date": None,
    }
