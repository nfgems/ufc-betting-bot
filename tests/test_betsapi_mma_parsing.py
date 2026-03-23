import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.betsapi_mma import (
    augment_features_with_betsapi_mma,
    inventory_saved_prematch_markets,
    load_saved_betsapi_mma_frames,
    normalize_mma_odds_summary_rows,
    normalize_bet365_prematch_market_rows,
    rebuild_betsapi_mma_history_from_raw,
    summarize_saved_betsapi_mma_backfill,
)
from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    UFCSTATS_REFRESH_ONLY_FEATURE_NAMES,
    get_betsapi_challenger_feature_columns,
    get_feature_family_columns,
    get_feature_columns_no_odds,
)


def _write_payload(path: Path, payload: dict, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata or {}, "payload": payload}, indent=2),
        encoding="utf-8",
    )


def test_normalize_bet365_prematch_market_rows_extracts_fight_lines():
    payload = {
        "success": 1,
        "results": [
            {
                "FI": "fi-1",
                "event_id": "evt-1",
                "main": {
                    "sp": {
                        "fight_lines": {
                            "id": "market-1",
                            "name": "Fight Lines",
                            "odds": [
                                {"id": "sel-a", "name": "1", "header": "To Win", "odds": "1.80", "handicap": ""},
                                {"id": "sel-b", "name": "2", "header": "To Win", "odds": "2.10", "handicap": ""},
                            ],
                        }
                    }
                }
            },
        ],
    }

    rows = normalize_bet365_prematch_market_rows(payload, snapshot_time_utc="2026-03-13T12:00:00Z")

    assert len(rows) == 2
    assert set(rows["market_name"]) == {"fight_lines"}
    assert set(rows["selection_header"]) == {"To Win"}
    assert rows["decimal_odds"].tolist() == pytest.approx([1.8, 2.1])


def test_normalize_bet365_prematch_market_rows_includes_top_level_others_blocks():
    payload = {
        "success": 1,
        "results": [
            {
                "FI": "fi-1",
                "event_id": "evt-1",
                "main": {"sp": {}},
                "others": [
                    {
                        "sp": {
                            "method_of_victory_double_chance": {
                                "id": "market-2",
                                "name": "Method of Victory Double Chance",
                                "odds": [
                                    {"id": "sel-a", "name": "KO, TKO, DQ or Decision", "header": "1", "odds": "1.74"},
                                    {"id": "sel-b", "name": "Submission or Decision", "header": "2", "odds": "2.40"},
                                ],
                            }
                        }
                    }
                ],
            },
        ],
    }

    rows = normalize_bet365_prematch_market_rows(payload, snapshot_time_utc="2026-03-13T12:00:00Z")

    assert len(rows) == 2
    assert set(rows["market_group"]) == {"others"}
    assert set(rows["market_name"]) == {"method_of_victory_double_chance"}


def test_normalize_bet365_prematch_market_rows_skips_intermediate_dicts_that_only_carry_ids():
    payload = {
        "success": 1,
        "results": [
            {
                "FI": "fi-1",
                "event_id": "evt-1",
                "main": {
                    "sp": {
                        "fight_lines": {
                            "id": "market-1",
                            "name": "Fight Lines",
                            "children": [
                                {"id": "sel-a", "name": "1", "header": "To Win", "odds": "1.80"},
                                {"id": "sel-b", "name": "2", "header": "To Win", "odds": "2.10"},
                            ],
                        }
                    }
                },
            },
        ],
    }

    rows = normalize_bet365_prematch_market_rows(payload, snapshot_time_utc="2026-03-13T12:00:00Z")

    assert len(rows) == 2
    assert set(rows["selection_id"]) == {"sel-a", "sel-b"}
    assert "market-1" not in set(rows["selection_id"])


def test_augment_features_with_betsapi_adds_columns_and_keeps_no_odds_clean(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "bet365_upcoming" / "20260313T120000Z_bet365_upcoming.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "fi-1",
                    "our_event_id": "evt-1",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                    "odds_updated_at": 1773356400,
                    "updated_at": 1773358200,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "bet365_prematch" / "20260313T120000Z_bet365_prematch.json",
        {
            "success": 1,
            "results": [
                {
                    "FI": "fi-1",
                    "event_id": "evt-1",
                    "main": {
                        "sp": {
                            "fight_lines": {
                                "id": "market-1",
                                "name": "Fight Lines",
                                "odds": [
                                    {"id": "sel-a", "name": "1", "header": "To Win", "odds": "1.80", "handicap": ""},
                                    {"id": "sel-b", "name": "2", "header": "To Win", "odds": "2.20", "handicap": ""},
                                ],
                            }
                        }
                    }
                }
            ],
        },
    )

    features_df = pd.DataFrame(
        [
                {
                    "event_date": pd.Timestamp("2026-03-13"),
                    "fighter_a": "Fighter A",
                    "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(features_df, raw_root=raw_root, save_artifacts=False)

    assert augmented.loc[0, "betsapi_snapshot_count"] == 1
    assert augmented.loc[0, "betsapi_a_implied_prob"] == pytest.approx(1 / 1.8)
    assert augmented.loc[0, "betsapi_b_implied_prob"] == pytest.approx(1 / 2.2)
    assert augmented.loc[0, "betsapi_market_disagreement_a"] == pytest.approx((1 / 1.8) - 0.53)
    assert bool(augmented.loc[0, "betsapi_match_found"]) is True

    no_odds_cols = get_feature_columns_no_odds(augmented)
    assert not any(column.startswith("betsapi_") for column in no_odds_cols)

    challenger_cols = get_betsapi_challenger_feature_columns(augmented)
    assert "betsapi_snapshot_count" in challenger_cols
    assert "betsapi_match_found" not in challenger_cols
    assert "betsapi_event_id" not in challenger_cols
    assert "betsapi_fi" not in challenger_cols
    assert "betsapi_league_name" not in challenger_cols
    assert "betsapi_scheduled_start_utc" not in challenger_cols
    assert all(pd.api.types.is_numeric_dtype(augmented[column]) for column in challenger_cols if column.startswith("betsapi_"))


def test_normalize_mma_odds_summary_rows_extracts_bookmaker_phase_rows():
    payload = {
        "success": 1,
        "results": {
            "Bet365": {
                "matching_dir": 1,
                "odds_update": {"start": "1773352800", "end": "1773367200"},
                "odds": {
                    "start": {
                        "1_1": {"id": "line-1", "home_od": "1.80", "away_od": "2.10", "add_time": "1773351000"},
                        "1_3": {"id": "line-3", "over_od": "1.90", "under_od": "1.90", "add_time": "1773351010"},
                    },
                    "end": {
                        "1_1": {"id": "line-2", "home_od": "1.55", "away_od": "2.55", "add_time": "1773367200"},
                    },
                },
            }
        },
    }

    rows = normalize_mma_odds_summary_rows(
        payload,
        event_id="evt-1",
        snapshot_time_utc="2026-03-13T12:00:00Z",
    )

    assert len(rows) == 3
    assert set(rows["market_phase"]) == {"start", "end"}
    assert set(rows["bookmaker"]) == {"Bet365"}
    assert rows.loc[rows["market_phase"] == "start", "home_odds"].iloc[0] == pytest.approx(1.8)
    ou_row = rows.loc[rows["market_key"] == "1_3"].iloc[0]
    assert ou_row["over_odds"] == pytest.approx(1.9)
    assert ou_row["under_odds"] == pytest.approx(1.9)


def test_augment_features_with_betsapi_uses_historical_opening_odds_summary(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "events_ended" / "20260313T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-1",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-1.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800", "end": "1773367200"},
                    "odds": {
                        "start": {
                            "162_1": {"id": "line-1", "home_od": "1.80", "away_od": "2.20", "add_time": "1773351000"},
                            "162_2": {"id": "spread", "home_od": "1.90", "away_od": "1.90", "handicap": "-1.5", "add_time": "1773351010"},
                            "162_3": {"id": "total-1", "over_od": "2.00", "under_od": "1.75", "handicap": "2.5", "add_time": "1773351020"},
                        },
                        "end": {
                            "162_1": {"id": "line-2", "home_od": "1.50", "away_od": "2.60", "add_time": "1773367200"},
                            "162_2": {"id": "spread-end", "home_od": "1.80", "away_od": "2.00", "handicap": "-2.5", "add_time": "1773367210"},
                            "162_3": {"id": "total-2", "over_od": "1.85", "under_od": "1.95", "handicap": "2.5", "add_time": "1773367220"},
                        },
                    },
                },
                "Pinnacle": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352860"},
                    "odds": {
                        "start": {
                            "162_1": {"id": "line-3", "home_od": "1.90", "away_od": "2.05", "add_time": "1773351060"},
                            "162_3": {"id": "total-3", "over_od": "1.95", "under_od": "1.80", "handicap": "2.5", "add_time": "1773351070"},
                        }
                    },
                },
            },
        },
        metadata={"event_id": "evt-1"},
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(features_df, raw_root=raw_root, save_artifacts=False)

    expected_a = ((1 / 1.80) + (1 / 1.90)) / 2
    expected_b = ((1 / 2.20) + (1 / 2.05)) / 2
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] == pytest.approx(expected_a)
    assert augmented.loc[0, "betsapi_hist_opening_b_implied_prob"] == pytest.approx(expected_b)
    assert augmented.loc[0, "betsapi_hist_diff_opening_implied_prob"] == pytest.approx(expected_a - expected_b)
    assert augmented.loc[0, "betsapi_hist_opening_bookmakers"] == 2
    assert augmented.loc[0, "betsapi_hist_market_disagreement_a"] == pytest.approx(expected_a - 0.53)
    assert augmented.loc[0, "betsapi_hist_m162_2_opening_home_implied_prob"] == pytest.approx(1 / 1.90)
    assert augmented.loc[0, "betsapi_hist_m162_2_end_home_implied_prob"] == pytest.approx(1 / 1.80)
    assert augmented.loc[0, "betsapi_hist_m162_2_handicap_move_abs"] == pytest.approx(1.0)
    expected_over = ((1 / 2.00) + (1 / 1.95)) / 2
    expected_under = ((1 / 1.75) + (1 / 1.80)) / 2
    assert augmented.loc[0, "betsapi_hist_m162_3_opening_over_implied_prob"] == pytest.approx(expected_over)
    assert augmented.loc[0, "betsapi_hist_m162_3_opening_under_implied_prob"] == pytest.approx(expected_under)
    assert augmented.loc[0, "betsapi_hist_m162_3_end_over_implied_prob"] == pytest.approx(1 / 1.85)
    assert augmented.loc[0, "betsapi_hist_m162_3_end_under_implied_prob"] == pytest.approx(1 / 1.95)
    assert augmented.loc[0, "betsapi_hist_m162_3_opening_handicap"] == pytest.approx(2.5)

    challenger_cols = get_betsapi_challenger_feature_columns(augmented)
    assert "betsapi_hist_opening_a_implied_prob" in challenger_cols
    assert "betsapi_hist_opening_b_implied_prob" in challenger_cols
    assert "betsapi_hist_opening_bookmakers" in challenger_cols
    assert "betsapi_hist_m162_2_end_home_implied_prob" in challenger_cols
    assert "betsapi_hist_m162_3_end_over_implied_prob" in challenger_cols

    production_betsapi_cols = get_feature_family_columns(augmented, "production_betsapi")
    assert "betsapi_hist_opening_a_implied_prob" in production_betsapi_cols
    assert "betsapi_hist_m162_2_end_home_implied_prob" in production_betsapi_cols
    assert "betsapi_hist_bookmaker_count" not in production_betsapi_cols
    assert "betsapi_snapshot_count" not in production_betsapi_cols

    production_betsapi_expanded_cols = get_feature_family_columns(
        augmented,
        "production_betsapi_expanded",
    )
    assert "betsapi_hist_bookmaker_count" in production_betsapi_expanded_cols


def test_augment_features_with_betsapi_matches_previous_utc_day_fight_key(tmp_path: Path):
    raw_root = tmp_path / "raw"
    event_time_utc = int(pd.Timestamp("2026-03-14T03:05:00Z").timestamp())

    _write_payload(
        raw_root / "events_ended" / "20260314T030500Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-previous-day",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": event_time_utc,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-previous-day.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773450000"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-1",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773446400",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-previous-day"},
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )

    assert bool(augmented.loc[0, "betsapi_match_found"]) is True
    assert augmented.loc[0, "betsapi_event_id"] == "evt-previous-day"
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] == pytest.approx(1 / 1.80)
    assert augmented.loc[0, "betsapi_hist_opening_b_implied_prob"] == pytest.approx(1 / 2.20)


def test_augment_features_with_betsapi_ignores_non_decimal_historical_moneyline_rows(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "events_ended" / "20260313T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-mixed-odds",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-mixed-odds.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-good",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773351000",
                            }
                        }
                    },
                },
                "VirginBet": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352810"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-bad",
                                "home_od": "0.63",
                                "away_od": "0.41",
                                "add_time": "1773351010",
                            }
                        }
                    },
                },
            },
        },
        metadata={"event_id": "evt-mixed-odds"},
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )
    frames = load_saved_betsapi_mma_frames(raw_root=raw_root)

    assert set(frames["summary_rows"]["bookmaker"]) == {"Bet365"}
    assert len(frames["summary_rows"]) == 1
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] == pytest.approx(1 / 1.80)
    assert augmented.loc[0, "betsapi_hist_opening_b_implied_prob"] == pytest.approx(1 / 2.20)
    assert augmented.loc[0, "betsapi_hist_opening_bookmakers"] == 1
    assert augmented.loc[0, "betsapi_hist_overround"] == pytest.approx((1 / 1.80) + (1 / 2.20) - 1.0)
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] < 1.0
    assert augmented.loc[0, "betsapi_hist_overround"] < 0.2


def test_augment_features_with_betsapi_prefers_shifted_signal_over_empty_exact_date_match(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "events_ended" / "20260314T030500Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-exact-empty",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": int(pd.Timestamp("2026-03-13T23:00:00Z").timestamp()),
                },
                {
                    "id": "evt-shifted-signal",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": int(pd.Timestamp("2026-03-14T03:05:00Z").timestamp()),
                },
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-shifted-signal.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773450000"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-1",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773446400",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-shifted-signal"},
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )

    assert bool(augmented.loc[0, "betsapi_match_found"]) is True
    assert augmented.loc[0, "betsapi_event_id"] == "evt-shifted-signal"
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] == pytest.approx(1 / 1.80)
    assert augmented.loc[0, "betsapi_hist_opening_b_implied_prob"] == pytest.approx(1 / 2.20)


def test_augment_features_with_betsapi_replaces_stale_betsapi_columns(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "events_ended" / "20260313T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-refresh",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-refresh.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-1",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773351000",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-refresh"},
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
                "betsapi_event_id": "stale-id",
                "betsapi_snapshot_count": 999,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )

    assert augmented.loc[0, "betsapi_event_id"] == "evt-refresh"
    assert augmented.loc[0, "betsapi_snapshot_count"] != 999
    assert not any(column.endswith("_x") or column.endswith("_y") for column in augmented.columns)


def test_load_saved_betsapi_mma_frames_ignores_noncanonical_summary_payloads_by_default(tmp_path: Path):
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "events_ended" / "20260313T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-canonical",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-canonical.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-1",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773351000",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-canonical"},
    )
    _write_payload(
        raw_root / "odds_summary" / "20260313T120000Z_odds_summary_event_evt-noncanonical.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-2",
                                "home_od": "1.60",
                                "away_od": "2.40",
                                "add_time": "1773351000",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-noncanonical"},
    )

    canonical_only = load_saved_betsapi_mma_frames(raw_root=raw_root)
    with_noncanonical = load_saved_betsapi_mma_frames(
        raw_root=raw_root,
        include_noncanonical_summary=True,
    )

    assert canonical_only["summary_rows"]["event_id"].tolist() == ["evt-canonical"]
    assert set(with_noncanonical["summary_rows"]["event_id"]) == {
        "evt-canonical",
        "evt-noncanonical",
    }


def test_rebuild_betsapi_mma_history_from_raw_refreshes_manifest_and_ignores_noncanonical_rows(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"

    _write_payload(
        raw_root / "events_ended" / "20260313T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-1",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "odds_summary" / "event_evt-1.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-1",
                                "home_od": "1.80",
                                "away_od": "2.20",
                                "add_time": "1773351000",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-1"},
    )
    _write_payload(
        raw_root / "odds_summary" / "20260313T120000Z_odds_summary_event_evt-ignored.json",
        {
            "success": 1,
            "results": {
                "Bet365": {
                    "matching_dir": 1,
                    "odds_update": {"start": "1773352800"},
                    "odds": {
                        "start": {
                            "162_1": {
                                "id": "line-2",
                                "home_od": "1.60",
                                "away_od": "2.40",
                                "add_time": "1773351000",
                            }
                        }
                    },
                }
            },
        },
        metadata={"event_id": "evt-ignored"},
    )

    processed_root.mkdir(parents=True, exist_ok=True)
    (processed_root / "backfill_manifest.json").write_text(
        json.dumps({"summary_rows": 99, "event_snapshots": 99, "event_rows": 99}),
        encoding="utf-8",
    )

    rebuild_result = rebuild_betsapi_mma_history_from_raw(
        raw_root=raw_root,
        processed_root=processed_root,
    )
    status = summarize_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    )

    assert rebuild_result["summary_rows"] == 1
    assert status["ready"] is True
    assert status["processed_row_counts"]["summary_rows"] == 1
    assert status["raw_summary_payload_files"] == 1
    assert status["raw_summary_payload_files_total"] == 2
    assert status["raw_summary_payload_files_noncanonical"] == 1
    assert status["manifest_matches_processed_counts"] is True
    assert status["manifest_is_fresh"] is True
    assert status["quality_errors"] == []


def test_inventory_saved_prematch_markets_counts_saved_rows(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_payload(
        raw_root / "bet365_prematch" / "20260313T120000Z_bet365_prematch.json",
        {
            "success": 1,
            "results": [
                {
                    "FI": "fi-1",
                    "event_id": "evt-1",
                    "main": {
                        "sp": {
                            "fight_lines": {
                                "id": "market-1",
                                "name": "Fight Lines",
                                "odds": [
                                    {"id": "sel-a", "name": "1", "header": "To Win", "odds": "1.80", "handicap": ""},
                                    {"id": "sel-b", "name": "2", "header": "To Win", "odds": "2.20", "handicap": ""},
                                ],
                            }
                        }
                    }
                }
            ],
        },
    )

    summary = inventory_saved_prematch_markets(raw_root=raw_root, save_csv=False)

    assert len(summary) == 1
    assert summary.loc[0, "market_name"] == "fight_lines"
    assert summary.loc[0, "selection_header"] == "To Win"
    assert summary.loc[0, "rows"] == 2


# ---------------------------------------------------------------------------
# T2 — ufcstats_betsapi_expanded uses refreshed base + challenger BetsAPI
# ---------------------------------------------------------------------------

def test_ufcstats_betsapi_expanded_uses_refreshed_base_plus_challenger_cols():
    row: dict = {
        "event_date": "2025-01-01",
        "fighter_a": "A",
        "fighter_b": "B",
        "target": 1,
        "diff_skill": 0.1,
        "a_roll_slpm": 3.5,
        "b_roll_slpm": 3.2,
        "a_num_fights": 8,
        "b_num_fights": 7,
        "a_implied_prob": 0.55,
        "b_implied_prob": 0.45,
        "diff_implied_prob": 0.10,
    }
    for col in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES:
        row[col] = 0.5
    for col in BETSAPI_CHALLENGER_FEATURE_NAMES:
        row[col] = 0.5
    df = pd.DataFrame([row])

    cols = get_feature_family_columns(df, "ufcstats_betsapi_expanded")

    # Must include refreshed-base columns
    for col in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES:
        assert col in cols, f"Missing refreshed col: {col}"
    # Must include all challenger BetsAPI columns
    for col in BETSAPI_CHALLENGER_FEATURE_NAMES:
        assert col in cols, f"Missing challenger col: {col}"
    # Must include production core columns
    assert "diff_skill" in cols


# ---------------------------------------------------------------------------
# T3 — previous-day fallback: exact-date wins tie when signal counts equal
# ---------------------------------------------------------------------------

def test_augment_features_with_betsapi_prefers_exact_date_when_both_have_equal_signal(tmp_path: Path):
    """When both exact-date and shifted-date rows have real odds with equal
    signal count, the exact-date match should win (priority 0 < priority 1)."""
    raw_root = tmp_path / "raw"

    # exact-date event: time in UTC = 2026-03-13 23:00 → date = 2026-03-13
    exact_time = int(pd.Timestamp("2026-03-13T23:00:00Z").timestamp())
    # shifted-date event: time in UTC = 2026-03-14 03:05 → date = 2026-03-14
    # shifted key would be 2026-03-13 (one day back)
    shifted_time = int(pd.Timestamp("2026-03-14T03:05:00Z").timestamp())

    _write_payload(
        raw_root / "events_ended" / "20260314T120000Z_events_ended_page_1.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "evt-exact",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": exact_time,
                },
                {
                    "id": "evt-shifted",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": shifted_time,
                },
            ],
        },
    )
    # Both events have real odds with equal signal (1 bookmaker each)
    for evt_id, home_od, away_od in [("evt-exact", "1.70", "2.30"), ("evt-shifted", "1.90", "2.00")]:
        _write_payload(
            raw_root / "odds_summary" / f"event_{evt_id}.json",
            {
                "success": 1,
                "results": {
                    "Bet365": {
                        "matching_dir": 1,
                        "odds_update": {"start": "1773352800"},
                        "odds": {
                            "start": {
                                "162_1": {
                                    "id": f"line-{evt_id}",
                                    "home_od": home_od,
                                    "away_od": away_od,
                                    "add_time": "1773351000",
                                }
                            }
                        },
                    }
                },
            },
            metadata={"event_id": evt_id},
        )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ]
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )

    # Exact-date match (evt-exact, home_od=1.70) should win
    assert augmented.loc[0, "betsapi_event_id"] == "evt-exact"
    assert augmented.loc[0, "betsapi_hist_opening_a_implied_prob"] == pytest.approx(1 / 1.70)


# ---------------------------------------------------------------------------
# Fix #1 test — positional assignment with non-RangeIndex
# ---------------------------------------------------------------------------

def test_augment_features_with_betsapi_correctly_aligns_with_non_range_index(tmp_path: Path):
    """When features_df has a non-RangeIndex, event metadata columns from the
    join must still be positionally aligned — no silent misalignment."""
    raw_root = tmp_path / "raw"

    _write_payload(
        raw_root / "bet365_upcoming" / "20260313T120000Z_bet365_upcoming.json",
        {
            "success": 1,
            "results": [
                {
                    "id": "fi-1",
                    "our_event_id": "evt-1",
                    "sport_id": 162,
                    "league": {"name": "UFC Fight Night"},
                    "home": {"name": "Fighter A"},
                    "away": {"name": "Fighter B"},
                    "time": 1773360000,
                    "odds_updated_at": 1773356400,
                    "updated_at": 1773358200,
                }
            ],
        },
    )
    _write_payload(
        raw_root / "bet365_prematch" / "20260313T120000Z_bet365_prematch.json",
        {
            "success": 1,
            "results": [
                {
                    "FI": "fi-1",
                    "event_id": "evt-1",
                    "main": {
                        "sp": {
                            "fight_lines": {
                                "id": "market-1",
                                "name": "Fight Lines",
                                "odds": [
                                    {"id": "sel-a", "name": "1", "header": "To Win", "odds": "1.80", "handicap": ""},
                                    {"id": "sel-b", "name": "2", "header": "To Win", "odds": "2.20", "handicap": ""},
                                ],
                            }
                        }
                    },
                }
            ],
        },
    )

    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-03-13"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "diff_stat": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 7,
                "a_implied_prob": 0.53,
                "b_implied_prob": 0.47,
                "diff_implied_prob": 0.06,
            }
        ],
        index=[42],  # Non-RangeIndex — this is the key scenario
    )

    augmented = augment_features_with_betsapi_mma(
        features_df,
        raw_root=raw_root,
        save_artifacts=False,
    )

    # Event metadata must be correctly positionally assigned despite the
    # non-RangeIndex input — use .iloc since merges reset the index.
    assert len(augmented) == 1
    assert bool(augmented.iloc[0]["betsapi_match_found"]) is True
    assert augmented.iloc[0]["betsapi_snapshot_count"] == 1
    assert augmented.iloc[0]["betsapi_a_implied_prob"] == pytest.approx(1 / 1.8)
    # Verify the fighter data is still correct (no row misalignment)
    assert augmented.iloc[0]["fighter_a"] == "Fighter A"
    assert augmented.iloc[0]["fighter_b"] == "Fighter B"
