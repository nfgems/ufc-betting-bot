import json
import pandas as pd
import pytest
import requests

from src.data import tennis_data
from src.data.tennis_data import (
    download_official_matches,
    derive_tournament_seed,
    load_tennis_matches,
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


def test_normalize_tennis_matches_preserves_player_orientation_after_filtered_rows():
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
            {
                "tour": "atp",
                "tourney_id": "2024-001",
                "tourney_name": "Test Open",
                "tourney_level": "A",
                "surface": "Hard",
                "draw_size": 32,
                "tourney_date": 20240103,
                "match_num": 3,
                "round": "R16",
                "best_of": 3,
                "minutes": 110,
                "score": "7-6 6-7 6-3",
                "winner_name": "Casper Ruud",
                "loser_name": "Holger Rune",
                "winner_id": 5,
                "loser_id": 6,
                "winner_rank": 8,
                "loser_rank": 7,
                "winner_rank_points": 4000,
                "loser_rank_points": 4200,
                "winner_age": 25.0,
                "loser_age": 20.0,
                "winner_hand": "R",
                "loser_hand": "R",
                "winner_ht": 183,
                "loser_ht": 188,
                "winner_seed": 8,
                "loser_seed": 7,
                "winner_entry": None,
                "loser_entry": None,
            },
        ]
    )

    normalized = normalize_tennis_matches(raw)

    assert len(normalized) == 2
    assert normalized.loc[0, "player_a"] == "Carlos Alcaraz"
    assert normalized.loc[0, "player_b"] == "Rafael Nadal"
    assert normalized.loc[0, "target"] == 0
    assert normalized.loc[1, "player_a"] == "Casper Ruud"
    assert normalized.loc[1, "player_b"] == "Holger Rune"
    assert normalized.loc[1, "target"] == 1


def test_parse_atp_archive_start_date_handles_same_month_and_cross_year_ranges():
    assert tennis_data._parse_atp_archive_start_date("6 - 12 January, 2025", 2025) == 20250106
    assert tennis_data._parse_atp_archive_start_date("30 December, 2024 - 5 January, 2025", 2025) == 20241230
    assert tennis_data._parse_atp_archive_start_date("27 April - 4 May, 2025", 2025) == 20250427


def test_map_wta_round_id_uses_draw_size_bracket():
    assert tennis_data._map_wta_round_id("1", 30) == "R32"
    assert tennis_data._map_wta_round_id("2", 30) == "R16"
    assert tennis_data._map_wta_round_id("1", 128) == "R128"
    assert tennis_data._map_wta_round_id("4", 128) == "R16"
    assert tennis_data._map_wta_round_id("1", 8, tournament_level="Finals", tournament_name="WTA Finals") == "RR"
    assert tennis_data._map_wta_round_id("Q", 128) == "QF"
    assert tennis_data._map_wta_round_id("S", 128) == "SF"
    assert tennis_data._map_wta_round_id("F", 128) == "F"


def test_infer_wta_draw_size_from_first_round_match_count():
    tournament_ref = {"draw_size": 0}
    matches = [
        {"RoundID": "1"},
        {"RoundID": "1"},
        {"RoundID": "1"},
        {"RoundID": "1"},
        {"RoundID": "2"},
        {"RoundID": "2"},
        {"RoundID": "Q"},
        {"RoundID": "S"},
        {"RoundID": "F"},
    ]

    assert tennis_data._infer_wta_draw_size(tournament_ref, matches) == 8


def test_infer_wta_winner_side_prefers_result_string_and_falls_back_to_flag_parity():
    result_string_match = {
        "ResultString": "D. Galfi d [6]A. Krueger 3-6,6-1,6-4",
        "Winner": "2",
        "PlayerNameFirstA": "Ashlyn",
        "PlayerNameLastA": "Krueger",
        "PlayerNameFirstB": "Dalma",
        "PlayerNameLastB": "Galfi",
    }
    parity_fallback_match = {
        "ResultString": "",
        "Winner": "5",
        "PlayerNameFirstA": "Laura",
        "PlayerNameLastA": "Siegemund",
        "PlayerNameFirstB": "Vera",
        "PlayerNameLastB": "Zvonareva",
    }

    assert tennis_data._infer_wta_winner_side(result_string_match) == "B"
    assert tennis_data._infer_wta_winner_side(parity_fallback_match) == "B"


def test_load_tennis_matches_uses_official_years_after_latest_sackmann(monkeypatch):
    official_calls: list[tuple[str, tuple[int, ...]]] = []
    sackmann_calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_available_years(tour, session=None):
        assert tour in {"atp", "wta"}
        return [2023, 2024]

    def fake_load_sackmann_matches(tour, years=None, force_download=False, session=None):
        sackmann_calls.append((tour, tuple(years or [])))
        return pd.DataFrame([{"tour": tour, "source_kind": "sackmann", "source_year": min(years or [0])}])

    def fake_load_official_matches(tour, years, force_download=False, session=None):
        official_calls.append((tour, tuple(years)))
        return pd.DataFrame([{"tour": tour, "source_kind": "official", "source_year": min(years or [0])}])

    monkeypatch.setattr(tennis_data, "list_available_sackmann_years", fake_available_years)
    monkeypatch.setattr(tennis_data, "load_sackmann_matches", fake_load_sackmann_matches)
    monkeypatch.setattr(tennis_data, "load_official_matches", fake_load_official_matches)
    monkeypatch.setattr(tennis_data, "_current_tennis_year", lambda: 2026)

    combined = load_tennis_matches(start_year=2024, end_year=2026)

    assert sackmann_calls == [("atp", (2024,)), ("wta", (2024,))]
    assert official_calls == [("atp", (2025, 2026)), ("wta", (2025, 2026))]
    assert set(combined["source_kind"]) == {"sackmann", "official"}


def test_download_official_matches_continues_after_fetch_error(monkeypatch, tmp_path):
    def fake_fetch_wta_official_year(year, session):
        if year == 2025:
            raise requests.HTTPError("429 Too Many Requests")
        return pd.DataFrame(
            [
                {
                    "tour": "wta",
                    "tourney_id": f"{year}-9999",
                    "tourney_name": "Test Event",
                    "tourney_level": "WTA 500",
                    "surface": "Hard",
                    "draw_size": 32,
                    "tourney_date": int(f"{year}0101"),
                    "match_num": 1,
                    "round": "R32",
                    "best_of": 3,
                    "score": "6-4 6-4",
                    "winner_name": "Player One",
                    "loser_name": "Player Two",
                    "winner_id": 1,
                    "loser_id": 2,
                }
            ]
        )

    monkeypatch.setattr(tennis_data, "_fetch_wta_official_year", fake_fetch_wta_official_year)
    monkeypatch.setattr(tennis_data, "_raw_dir_for_tour", lambda tour: tmp_path)

    paths = download_official_matches("wta", [2025, 2026], force=True, session=requests.Session())

    assert [path.name for path in paths] == ["wta_matches_2026_official.csv"]


def test_discover_wta_tournaments_uses_cached_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "wta_tournaments_2025_official_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": tennis_data.WTA_MANIFEST_SCHEMA_VERSION,
                "year": 2025,
                "tournaments": [
                    {
                        "tournament_id": "1001",
                        "slug": "test-event",
                        "year": 2025,
                        "tourney_name": "Test Event",
                        "tourney_level": "WTA 500",
                        "surface": "Hard",
                        "draw_size": 32,
                        "tourney_date": 20250101,
                        "tourney_end_date": 20250107,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tennis_data, "WTA_RAW_DIR", tmp_path)
    monkeypatch.setattr(
        tennis_data,
        "_get_with_retries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be used when manifest exists")),
    )

    tournaments = tennis_data._discover_wta_tournaments(2025, requests.Session())

    assert tournaments == [
        {
            "tournament_id": "1001",
            "slug": "test-event",
            "year": 2025,
            "tourney_name": "Test Event",
            "tourney_level": "WTA 500",
            "surface": "Hard",
            "draw_size": 32,
            "tourney_date": 20250101,
            "tourney_end_date": 20250107,
        }
    ]


def test_discover_wta_tournaments_resumes_from_manifest_progress(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

    call_log: list[str] = []

    def first_run_get_with_retries(session, url, timeout=60, max_attempts=5):
        call_log.append(url)
        if "page=0" in url:
            return FakeResponse({"pageInfo": {"numEntries": 250}})
        if "page=2" in url:
            return FakeResponse(
                {
                    "content": [
                        {
                            "year": 2025,
                            "tournamentGroup": {"id": 1001, "name": "Event A"},
                            "title": "Event A - City",
                            "level": "WTA 500",
                            "surface": "Hard",
                            "singlesDrawSize": 32,
                            "startDate": "2025-01-01",
                            "endDate": "2025-01-07",
                        }
                    ]
                }
            )
        if "page=1" in url:
            raise requests.HTTPError("429 Too Many Requests")
        raise AssertionError(f"Unexpected URL on first run: {url}")

    monkeypatch.setattr(tennis_data, "WTA_RAW_DIR", tmp_path)
    monkeypatch.setattr(tennis_data, "_get_with_retries", first_run_get_with_retries)

    with pytest.raises(requests.HTTPError):
        tennis_data._discover_wta_tournaments(2025, requests.Session())

    progress_path = tmp_path / "wta_tournaments_2025_official_manifest_progress.json"
    assert progress_path.exists()
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["next_page"] == 1
    assert [item["tournament_id"] for item in progress_payload["tournaments"]] == ["1001"]

    def second_run_get_with_retries(session, url, timeout=60, max_attempts=5):
        call_log.append(url)
        if "page=1" in url:
            return FakeResponse(
                {
                    "content": [
                        {
                            "year": 2025,
                            "tournamentGroup": {"id": 1002, "name": "Event B"},
                            "title": "Event B - City",
                            "level": "WTA 250",
                            "surface": "Clay",
                            "singlesDrawSize": 32,
                            "startDate": "2025-01-08",
                            "endDate": "2025-01-14",
                        }
                    ]
                }
            )
        if "page=0" in url:
            return FakeResponse(
                {
                    "content": [
                        {
                            "year": 2024,
                            "tournamentGroup": {"id": 999, "name": "Old Event"},
                            "title": "Old Event - City",
                            "level": "WTA 250",
                            "surface": "Hard",
                            "singlesDrawSize": 32,
                            "startDate": "2024-12-01",
                            "endDate": "2024-12-07",
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL on second run: {url}")

    monkeypatch.setattr(tennis_data, "_get_with_retries", second_run_get_with_retries)

    tournaments = tennis_data._discover_wta_tournaments(2025, requests.Session())

    assert [item["tournament_id"] for item in tournaments] == ["1001", "1002"]
    assert not progress_path.exists()
    assert (tmp_path / "wta_tournaments_2025_official_manifest.json").exists()
    assert sum("page=0" in url for url in call_log) == 2


def test_download_official_matches_resumes_wta_progress(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

    tournaments = [
        {
            "tournament_id": "1001",
            "slug": "event-a",
            "year": 2025,
            "tourney_name": "Event A",
            "tourney_level": "WTA 500",
            "surface": "Hard",
            "draw_size": 32,
            "tourney_date": 20250101,
            "tourney_end_date": 20250107,
        },
        {
            "tournament_id": "1002",
            "slug": "event-b",
            "year": 2025,
            "tourney_name": "Event B",
            "tourney_level": "WTA 250",
            "surface": "Clay",
            "draw_size": 32,
            "tourney_date": 20250108,
            "tourney_end_date": 20250114,
        },
    ]
    payload_a = {
        "matches": [
            {
                "DrawMatchType": "S",
                "DrawLevelType": "M",
                "MatchState": "F",
                "PlayerNameFirstA": "Player",
                "PlayerNameLastA": "One",
                "PlayerNameFirstB": "Player",
                "PlayerNameLastB": "Two",
                "PlayerIDA": 1,
                "PlayerIDB": 2,
                "SeedA": "",
                "SeedB": "",
                "EntryTypeA": "",
                "EntryTypeB": "",
                "RoundID": "1",
                "ScoreString": "6-4,6-4",
                "ResultString": "P. One d P. Two 6-4,6-4",
                "Winner": "2",
                "MatchTimeStamp": "2025-01-02T10:00:00Z",
                "MatchID": "M1",
            }
        ]
    }
    payload_b = {
        "matches": [
            {
                "DrawMatchType": "S",
                "DrawLevelType": "M",
                "MatchState": "F",
                "PlayerNameFirstA": "Player",
                "PlayerNameLastA": "Three",
                "PlayerNameFirstB": "Player",
                "PlayerNameLastB": "Four",
                "PlayerIDA": 3,
                "PlayerIDB": 4,
                "SeedA": "",
                "SeedB": "",
                "EntryTypeA": "",
                "EntryTypeB": "",
                "RoundID": "1",
                "ScoreString": "6-3,6-3",
                "ResultString": "P. Four d P. Three 6-3,6-3",
                "Winner": "3",
                "MatchTimeStamp": "2025-01-09T10:00:00Z",
                "MatchID": "M2",
            }
        ]
    }

    call_log: list[str] = []

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        call_log.append(url)
        if url.endswith("/1001/2025/matches"):
            return FakeResponse(payload_a)
        if url.endswith("/1002/2025/matches") and sum(item.endswith("/1002/2025/matches") for item in call_log) == 1:
            raise requests.HTTPError("429 Too Many Requests")
        if url.endswith("/1002/2025/matches"):
            return FakeResponse(payload_b)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(tennis_data, "WTA_RAW_DIR", tmp_path)
    monkeypatch.setattr(tennis_data, "_discover_wta_tournaments", lambda year, session: tournaments)
    monkeypatch.setattr(tennis_data, "_get_with_retries", fake_get_with_retries)
    monkeypatch.setattr(tennis_data, "OFFICIAL_REQUEST_PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(tennis_data, "_today_yyyymmdd", lambda: 20260101)

    first_paths = download_official_matches("wta", [2025], force=True, session=requests.Session())
    progress_state_path = tmp_path / "wta_matches_2025_official_progress.json"
    progress_rows_path = tmp_path / "wta_matches_2025_official_progress.csv"

    assert first_paths == []
    assert progress_state_path.exists()
    assert progress_rows_path.exists()

    progress_state = json.loads(progress_state_path.read_text(encoding="utf-8"))
    assert progress_state["completed_tournament_ids"] == ["1001"]

    second_paths = download_official_matches("wta", [2025], force=True, session=requests.Session())

    assert [path.name for path in second_paths] == ["wta_matches_2025_official.csv"]
    final_frame = pd.read_csv(second_paths[0])
    assert set(final_frame["tourney_id"]) == {"2025-1001", "2025-1002"}
    assert not progress_state_path.exists()
    assert not progress_rows_path.exists()
    assert sum(url.endswith("/1001/2025/matches") for url in call_log) == 1
    assert sum(url.endswith("/1002/2025/matches") for url in call_log) == 2


def test_normalize_tennis_matches_accepts_official_rows_with_missing_optional_columns():
    raw = pd.DataFrame(
        [
            {
                "tour": "wta",
                "tourney_id": "2026-2014",
                "tourney_name": "Adelaide",
                "tourney_level": "WTA 500",
                "surface": "Hard",
                "draw_size": 30,
                "tourney_date": 20260112,
                "match_num": 1,
                "round": "R32",
                "best_of": 3,
                "score": "5-7 6-3 6-2",
                "winner_name": "Beatriz Haddad Maia",
                "loser_name": "Victoria Mboko",
                "winner_id": 1001,
                "loser_id": 1002,
                "winner_seed": 1,
                "loser_seed": pd.NA,
                "winner_entry": pd.NA,
                "loser_entry": "WC",
            }
        ]
    )

    normalized = normalize_tennis_matches(raw)

    assert len(normalized) == 1
    assert normalized.loc[0, "winner"] == "Beatriz Haddad Maia"
    assert normalized.loc[0, "player_a"] == "Beatriz Haddad Maia"
    assert normalized.loc[0, "player_b"] == "Victoria Mboko"
    assert normalized.loc[0, "target"] == 1
