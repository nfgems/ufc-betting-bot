import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src import bot
from src.data.name_utils import normalize_cross_source_name


def _make_repo_local_tmp_dir() -> Path:
    path = Path.cwd() / "data" / f"bot-live-context-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _base_live_features() -> dict:
    return {
        "a_num_fights": 5,
        "b_num_fights": 6,
        "a_ko_rate": 0.12,
        "b_ko_rate": 0.18,
        "a_sub_rate": 0.04,
        "b_sub_rate": 0.03,
        "a_dec_rate": 0.54,
        "b_dec_rate": 0.49,
        "a_roll_slpm": 3.8,
        "b_roll_slpm": 4.1,
        "a_roll_kd": 0.4,
        "b_roll_kd": 0.5,
        "a_roll_sub_avg": 0.2,
        "b_roll_sub_avg": 0.1,
        "a_roll_td_avg": 1.5,
        "b_roll_td_avg": 1.2,
        "a_total_rounds": 18.0,
        "b_total_rounds": 20.0,
        "a_roll_str_def": 0.57,
        "b_roll_str_def": 0.54,
        "a_roll_td_def": 0.69,
        "b_roll_td_def": 0.64,
        "a_roll_sapm": 2.9,
        "b_roll_sapm": 3.2,
        "a_wins": 8.0,
        "b_wins": 10.0,
        "a_losses": 2.0,
        "b_losses": 3.0,
        "a_draws": 0.0,
        "b_draws": 0.0,
        "a_win_pct": 0.8,
        "b_win_pct": 0.7692,
        "a_current_win_streak": 2.0,
        "b_current_win_streak": 1.0,
        "a_lose_streak": 0.0,
        "b_lose_streak": 0.0,
        "a_days_since_last_fight": 180.0,
        "b_days_since_last_fight": 210.5,
        "a_cage_rust": 0.0,
        "b_cage_rust": 0.0,
    }


def _fake_model_result(artifact_path: Path, *, feature_cols: list[str] | None = None) -> dict:
    cols = list(feature_cols or [])
    return {
        "feature_cols": cols,
        "col_medians": np.zeros(len(cols)),
        "feature_importance": {},
        "raw_model": None,
        "artifact_path": str(artifact_path),
        "training_spec": {
            "name": "unit-test-spec",
            "feature_cols": cols,
        },
    }


def _cached_prediction_row(
    fight: dict,
    *,
    runtime_signature: dict,
    generated_at: str,
    prob_a: float = 0.61,
    prob_b: float = 0.39,
    confidence: float = 0.61,
    a_market_prob: float | None = None,
    b_market_prob: float | None = None,
    line_movement: float | None = None,
) -> dict:
    a_market = fight["a_fair_prob_avg"] if a_market_prob is None else a_market_prob
    b_market = fight["b_fair_prob_avg"] if b_market_prob is None else b_market_prob
    features = _base_live_features()
    row = {
        "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
        "fighter_a": fight["fighter_a"],
        "fighter_b": fight["fighter_b"],
        "prob_a": prob_a,
        "prob_b": prob_b,
        "confidence": confidence,
        "event_id": fight.get("event_id", ""),
        "event_date": fight["commence_time"],
        "a_market_prob": a_market,
        "b_market_prob": b_market,
        "no_odds_prob_a": None,
        "no_odds_prob_b": None,
        "a_num_fights": 5,
        "b_num_fights": 6,
        "shap_values": [],
        "shap_base_value": None,
        "feature_highlights": [],
        "low_experience": False,
        "method_stats": {
            "a_ko_rate": 0.12,
            "b_ko_rate": 0.18,
        },
        "fighter_context": {
            "a_wins": 8,
            "b_wins": 10,
        },
        "pair_key": bot._live_fight_pair_key(fight["fighter_a"], fight["fighter_b"]),
        "cache_key": bot._prediction_cache_key(fight),
        "prediction_generated_at": generated_at,
        "odds_snapshot": {
            "a_fair_prob_avg": a_market,
            "b_fair_prob_avg": b_market,
        },
        "event_context_snapshot": {
            "event_id": fight.get("event_id", ""),
            "commence_time": bot._prediction_commence_token(fight["commence_time"]),
            "weight_class": "Bantamweight",
            "num_rounds": 3,
            "is_title_bout": False,
            "is_empty_arena": False,
        },
        "runtime_signature": runtime_signature,
        "operator_features": features,
        "operator_provenance": {
            "bundle_id": "unit-bundle",
            "model_spec_name": "unit-test-spec",
        },
    }
    if line_movement is not None:
        row["line_movement"] = line_movement
        row["line_is_sharp"] = 0
        row["line_steam_move"] = 0
    return row


def test_resolve_live_fight_counts_prefers_live_feature_counts():
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": 17,
            "b_num_fights": "9",
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda _: 0,
    )

    assert counts == (17, 9)


def test_resolve_live_fight_counts_falls_back_when_live_counts_missing():
    fallback_counts = {
        "Israel Adesanya": 18,
        "Joseph Pyfer": 0,
    }
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": None,
            "b_num_fights": float("nan"),
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda name: fallback_counts[name],
    )

    assert counts == (18, 0)


def test_infer_weight_class_from_history_falls_back_to_raw_history_when_processed_schema_invalid(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (processed_dir / "fights_cleaned.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        assert bot._infer_weight_class_from_history("Ricky Simon", "Adrian Yanez") == "Bantamweight"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_uses_raw_history_when_processed_history_missing(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Bantamweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_falls_back_to_near_term_ufc_lookup_when_history_missing(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup._lookup_processed_fighter",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda fighter_name: f"http://example.test/{fighter_name.replace(' ', '-').lower()}",
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.scrape_fighter_fights",
            lambda *args, **kwargs: [
                {"event_date": "2025-09-01", "weight_class": "Bantamweight"},
            ],
        )

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "event_date": "March 28, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Bantamweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_near_term_lookup_requires_both_fighters_to_resolve(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nMarcin Tybura\nTyrell Fortune\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup._lookup_processed_fighter",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda fighter_name: "http://example.test/marcin-tybura" if fighter_name == "Marcin Tybura" else None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.scrape_fighter_fights",
            lambda *args, **kwargs: [
                {"event_date": "2025-09-01", "weight_class": "Heavyweight"},
            ],
        )

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Marcin Tybura",
                "fighter_b": "Tyrell Fortune",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "event_date": "March 28, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_load_local_ufc_roster_names_unions_official_roster_artifact(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "ufc_active_roster_official.csv"

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\n",
            encoding="utf-8",
        )
        official_path.write_text(
            "\n".join(
                [
                    "official_name,slug_name,alternate_slug_names,ufcstats_name",
                    "Nariman Abbassov,nariman abbassov,nariman abbasov,Nariman Abbasov",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        roster_names = bot._load_local_ufc_roster_names()

        assert normalize_cross_source_name("Ricky Simon") in roster_names
        assert "nariman abbassov" in roster_names
        assert "nariman abbasov" in roster_names
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_matches_official_slug_aliases(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "ufc_active_roster_official.csv"
        official_path.write_text(
            "\n".join(
                [
                    "official_name,profile_name,slug_name,alternate_slug_names,ufcstats_name",
                    "Patricio Pitbull,Patricio Pitbull,patricio pitbull freire,patricio pitbull freire,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-12T00:40:00+00:00",
                "fighter_a": "Aaron Pico",
                "fighter_b": "Patricio Pitbull",
            },
            [
                {
                    "event_id": "evt-1",
                    "commence_time": "2026-04-12T00:40:00+00:00",
                    "event_date": "April 11, 2026",
                    "fighter_a": "Patricio Freire",
                    "fighter_b": "Aaron Pico",
                    "weight_class": "Featherweight",
                    "is_main_event": False,
                    "is_title_bout": False,
                    "num_rounds": 3,
                }
            ],
            allow_off_card_history_fallback=False,
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Featherweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_matches_initialed_name_variants(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "missing_roster.csv"

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-2",
                "commence_time": "2026-04-18T22:00:00+00:00",
                "fighter_a": "J.J. Aldrich",
                "fighter_b": "Jamey-Lyn Horth",
            },
            [
                {
                    "event_id": "evt-2",
                    "commence_time": "2026-04-18T22:00:00+00:00",
                    "event_date": "April 18, 2026",
                    "fighter_a": "JJ Aldrich",
                    "fighter_b": "Jamey-Lyn Horth",
                    "weight_class": "Women's Flyweight",
                    "is_main_event": False,
                    "is_title_bout": False,
                    "num_rounds": 3,
                }
            ],
            allow_off_card_history_fallback=False,
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Women's Flyweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_skips_near_term_lookup_for_far_future_dates(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected UFCStats lookup")),
        )

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-01T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-05-01T20:00:00+00:00",
                    "event_date": "May 1, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_blocks_recent_prior_matchup_without_official_context(
    monkeypatch,
    caplog,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        with caplog.at_level(logging.INFO):
            event_context = bot._resolve_live_event_context(
                {
                    "event_id": "evt-1",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "fighter_a": "Charles Johnson",
                    "fighter_b": "Bruno Silva",
                },
                [],
            )

        assert event_context is None
        assert any(
            record.levelno == logging.INFO and "Refusing fallback live context" in record.message
            for record in caplog.records
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_allows_far_future_rematch_history_fallback(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-20T20:00:00+00:00",
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
            },
            [],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Flyweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_blocks_off_card_history_fallback_in_strict_mode(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-20T20:00:00+00:00",
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
            },
            [],
            allow_off_card_history_fallback=False,
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_caches_predictions_with_resolved_live_context(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        captured = {}

        def fake_build_fight_features(*args, **kwargs):
            captured["fighter_a"] = args[0]
            captured["fighter_b"] = args[1]
            captured["weight_class"] = kwargs["weight_class"]
            return {
                "a_num_fights": 5,
                "b_num_fights": 6,
                "a_ko_rate": 0.0,
                "b_ko_rate": 0.0,
                "a_sub_rate": 0.0,
                "b_sub_rate": 0.0,
                "a_dec_rate": 0.0,
                "b_dec_rate": 0.0,
                "a_roll_slpm": 0.0,
                "b_roll_slpm": 0.0,
                "a_roll_kd": 0.0,
                "b_roll_kd": 0.0,
                "a_roll_sub_avg": 0.0,
                "b_roll_sub_avg": 0.0,
                "a_roll_td_avg": 0.0,
                "b_roll_td_avg": 0.0,
                "a_total_rounds": 0.0,
                "b_total_rounds": 0.0,
            }

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.model.predict.predict_fight",
            lambda *_args, **_kwargs: {"prob_a": 0.61, "prob_b": 0.39, "confidence": 0.61},
        )
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        cache_path = logs_dir / "predictions_cache.json"
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert captured["fighter_a"] == "Ricky Simon"
        assert captured["fighter_b"] == "Adrian Yanez"
        assert captured["weight_class"] == "Bantamweight"
        assert len(payload["predictions"]) == 1
        assert payload["predictions"][0]["fighter_a"] == "Ricky Simon"
        assert payload["predictions"][0]["fighter_b"] == "Adrian Yanez"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_serializes_nan_fighter_context_without_crashing(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        def fake_build_fight_features(*args, **kwargs):
            return {
                "a_num_fights": 5,
                "b_num_fights": 6,
                "a_ko_rate": float("nan"),
                "b_ko_rate": 0.18,
                "a_sub_rate": 0.0,
                "b_sub_rate": 0.0,
                "a_dec_rate": 0.0,
                "b_dec_rate": 0.0,
                "a_roll_slpm": 0.0,
                "b_roll_slpm": 0.0,
                "a_roll_kd": 0.0,
                "b_roll_kd": 0.0,
                "a_roll_sub_avg": 0.0,
                "b_roll_sub_avg": 0.0,
                "a_roll_td_avg": 0.0,
                "b_roll_td_avg": 0.0,
                "a_total_rounds": 0.0,
                "b_total_rounds": 0.0,
                "a_roll_str_def": 0.0,
                "b_roll_str_def": 0.0,
                "a_roll_td_def": 0.0,
                "b_roll_td_def": 0.0,
                "a_roll_sapm": 0.0,
                "b_roll_sapm": 0.0,
                "a_wins": float("nan"),
                "b_wins": 10.0,
                "a_losses": 2.0,
                "b_losses": 3.0,
                "a_draws": 0.0,
                "b_draws": 1.0,
                "a_win_pct": float("nan"),
                "b_win_pct": 0.7692,
                "a_current_win_streak": float("nan"),
                "b_current_win_streak": 2.0,
                "a_lose_streak": 1.0,
                "b_lose_streak": 0.0,
                "a_days_since_last_fight": float("nan"),
                "b_days_since_last_fight": 210.5,
                "a_cage_rust": float("nan"),
                "b_cage_rust": 0.0,
            }

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.model.predict.predict_fight",
            lambda *_args, **_kwargs: {"prob_a": 0.61, "prob_b": 0.39, "confidence": 0.61},
        )
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        prediction = payload["predictions"][0]

        assert prediction["fighter_context"]["a_wins"] is None
        assert prediction["fighter_context"]["a_win_pct"] is None
        assert prediction["fighter_context"]["a_current_win_streak"] is None
        assert prediction["fighter_context"]["a_days_since_last_fight"] is None
        assert prediction["fighter_context"]["a_cage_rust"] is None
        assert prediction["fighter_context"]["b_wins"] == 10
        assert prediction["fighter_context"]["b_win_pct"] == 0.7692
        assert prediction["fighter_context"]["b_days_since_last_fight"] == 210.5
        assert prediction["method_stats"]["a_ko_rate"] is None
        assert prediction["method_stats"]["b_ko_rate"] == 0.18
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_writes_empty_cache_when_all_fights_are_skipped(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 19, 55, tzinfo=timezone.utc))
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        cache_path = logs_dir / "predictions_cache.json"
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload["predictions"] == []
        assert isinstance(payload["timestamp"], str) and payload["timestamp"]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_skips_off_card_fight_before_line_and_injury_checks(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        fight = {
            "event_id": "stale-odds-event",
            "commence_time": "2026-06-07T00:45:00Z",
            "fighter_a": "Bryce Mitchell",
            "fighter_b": "Santiago Ponzinibbio",
            "a_fair_prob_avg": 0.56,
            "b_fair_prob_avg": 0.44,
            "num_bookmakers": 1,
        }

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame([fight])

        line_calls: list[tuple] = []
        injury_calls: list[tuple] = []

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc))
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts_for_fights", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(bot, "_resolve_live_event_context", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_missing_live_event_context_reason",
            lambda *_args, **_kwargs: "not on any upcoming UFC card",
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *args, **kwargs: line_calls.append((args, kwargs)) or {},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *args, **kwargs: injury_calls.append((args, kwargs)) or {"suspected": False},
        )
        monkeypatch.setattr("src.polymarket.markets.get_ufc_fight_markets", lambda: pd.DataFrame())
        monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", lambda **_kwargs: {"total_orders": 0})

        result = bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert result == {"status": "idle", "reason": "no_executable_opportunities"}
        assert line_calls == []
        assert injury_calls == []
        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert payload["predictions"] == []
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_reuses_cached_predictions_without_rebuilding(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        fight = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        cached_row = _cached_prediction_row(
            fight,
            runtime_signature=runtime_signature,
            generated_at="2026-03-28T18:00:00+00:00",
            a_market_prob=0.55,
            b_market_prob=0.45,
            line_movement=0.0,
        )
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": [cached_row],
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            **fight,
                            "a_fair_prob_avg": 0.56,
                            "b_fair_prob_avg": 0.44,
                        }
                    ]
                )

        build_calls: list[int] = []
        predict_calls: list[int] = []
        captured: dict = {}

        def fake_build_fight_features(*_args, **_kwargs):
            build_calls.append(1)
            return _base_live_features(), {"fighter_a_source": "build", "fighter_b_source": "build"}

        def fake_predict_fight(*_args, **_kwargs):
            predict_calls.append(1)
            return {"prob_a": 0.63, "prob_b": 0.37, "confidence": 0.63}

        def fake_run_duo_traders(**kwargs):
            captured["predictions"] = kwargs["predictions"].copy()
            captured["features_by_fight"] = dict(kwargs["features_by_fight"])
            captured["provenance_by_fight"] = dict(kwargs["provenance_by_fight"])
            return {"total_orders": 0}

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.odds_client.OddsClient",
            FakeOddsClient,
        )
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr("src.model.predict.predict_fight", fake_predict_fight)
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.01, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", fake_run_duo_traders)

        result = bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert result == {"status": "ok", "total_orders": 0}
        assert build_calls == []
        assert predict_calls == []
        assert captured["features_by_fight"]["Ricky Simon|Adrian Yanez"] == cached_row["operator_features"]
        assert captured["provenance_by_fight"]["Ricky Simon|Adrian Yanez"] == cached_row["operator_provenance"]
        assert float(captured["predictions"].iloc[0]["a_market_prob"]) == 0.56
        assert float(captured["predictions"].iloc[0]["line_movement"]) == 0.01

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        prediction = payload["predictions"][0]
        assert prediction["prediction_generated_at"] == "2026-03-28T18:00:00+00:00"
        assert prediction["a_market_prob"] == 0.56
        assert prediction["odds_snapshot"]["a_fair_prob_avg"] == 0.56
        assert prediction["line_movement"] == 0.01
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_keeps_full_cache_visible_while_refreshing_single_fight(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        fight_one = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        fight_two = {
            "event_id": "evt-2",
            "commence_time": "2026-03-28T21:00:00Z",
            "fighter_a": "Rob Font",
            "fighter_b": "Marlon Vera",
            "a_fair_prob_avg": 0.40,
            "b_fair_prob_avg": 0.60,
            "num_bookmakers": 8,
        }
        cached_predictions = [
            _cached_prediction_row(
                fight_one,
                runtime_signature=runtime_signature,
                generated_at="2026-03-28T18:00:00+00:00",
                a_market_prob=0.55,
                b_market_prob=0.45,
                line_movement=0.0,
            ),
            _cached_prediction_row(
                fight_two,
                runtime_signature=runtime_signature,
                generated_at="2026-03-28T18:00:00+00:00",
                a_market_prob=0.40,
                b_market_prob=0.60,
                line_movement=0.0,
            ),
        ]
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": cached_predictions,
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            **fight_one,
                            "a_fair_prob_avg": 0.56,
                            "b_fair_prob_avg": 0.44,
                        },
                        {
                            **fight_two,
                            "a_fair_prob_avg": 0.47,
                            "b_fair_prob_avg": 0.53,
                        },
                    ]
                )

        build_calls: list[str] = []
        predict_calls: list[str] = []
        context_loads: list[int] = []
        cache_write_lengths: list[int] = []

        original_write_text = Path.write_text

        def recording_write_text(path_obj, data, *args, **kwargs):
            if path_obj.name == "predictions_cache.json.tmp":
                cache_write_lengths.append(len(json.loads(data)["predictions"]))
            return original_write_text(path_obj, data, *args, **kwargs)

        def fake_build_fight_features(fighter_a, fighter_b, **_kwargs):
            build_calls.append(f"{fighter_a}|{fighter_b}")
            return _base_live_features(), {"fighter_a_source": "refresh", "fighter_b_source": "refresh"}

        def fake_predict_fight(features, **_kwargs):
            predict_calls.append("predict")
            return {"prob_a": 0.58, "prob_b": 0.42, "confidence": 0.58}

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(Path, "write_text", recording_write_text)
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr("src.model.predict.predict_fight", fake_predict_fight)
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: context_loads.append(1) or [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.02, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ufc-card"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert build_calls == ["Rob Font|Marlon Vera"]
        assert predict_calls == ["predict"]
        assert context_loads == [1]
        assert cache_write_lengths
        assert all(length == 2 for length in cache_write_lengths)

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert len(payload["predictions"]) == 2
        refreshed = next(row for row in payload["predictions"] if row["fighter_a"] == "Rob Font")
        assert refreshed["prediction_generated_at"] != "2026-03-28T18:00:00+00:00"
        assert refreshed["a_market_prob"] == 0.47
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_prunes_removed_cached_predictions(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        current_fight = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        removed_fight = {
            "event_id": "evt-2",
            "commence_time": "2026-03-28T21:00:00Z",
            "fighter_a": "Rob Font",
            "fighter_b": "Marlon Vera",
            "a_fair_prob_avg": 0.40,
            "b_fair_prob_avg": 0.60,
            "num_bookmakers": 8,
        }
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": [
                        _cached_prediction_row(
                            current_fight,
                            runtime_signature=runtime_signature,
                            generated_at="2026-03-28T18:00:00+00:00",
                        ),
                        _cached_prediction_row(
                            removed_fight,
                            runtime_signature=runtime_signature,
                            generated_at="2026-03-28T18:00:00+00:00",
                        ),
                    ],
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame([current_fight])

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.odds_client.OddsClient",
            FakeOddsClient,
        )
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.01, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ufc-card"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert len(payload["predictions"]) == 1
        assert payload["predictions"][0]["fighter_a"] == "Ricky Simon"
        assert payload["predictions"][0]["fighter_b"] == "Adrian Yanez"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_load_live_event_contexts_reuses_matching_cached_card_within_ttl(monkeypatch):
    from src.data import live_monitor

    cached_contexts = [
        {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00+00:00",
            "event_date": "March 28, 2026",
        }
    ]

    def _fail_collect():
        raise RuntimeError("boom")

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _fail_collect)
    monkeypatch.setattr(bot, "_LAST_GOOD_LIVE_EVENT_CONTEXTS", (100.0, ("event:evt-1",), cached_contexts))
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
            }
        ]
    )

    assert result == cached_contexts


def test_load_live_event_contexts_reuses_cached_card_subset_within_ttl(monkeypatch):
    from src.data import live_monitor

    cached_contexts = [
        {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00+00:00",
            "event_date": "March 28, 2026",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
            "weight_class": "Lightweight",
        }
    ]
    collect_calls = {"count": 0}

    def _empty_collect():
        collect_calls["count"] += 1
        return []

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _empty_collect)
    monkeypatch.setattr(bot, "_LAST_GOOD_LIVE_EVENT_CONTEXTS", (100.0, ("event:evt-1",), cached_contexts))
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
            },
            {
                "event_id": "evt-2",
                "commence_time": "2026-03-28T21:00:00+00:00",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
            },
        ]
    )

    assert collect_calls["count"] == 1
    assert result == cached_contexts


def test_load_live_event_contexts_rejects_mismatched_cached_card(monkeypatch):
    from src.data import live_monitor

    def _fail_collect():
        raise RuntimeError("boom")

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _fail_collect)
    monkeypatch.setattr(
        bot,
        "_LAST_GOOD_LIVE_EVENT_CONTEXTS",
        (
            100.0,
            ("event:evt-1",),
            [{"event_id": "evt-1", "commence_time": "2026-03-28T20:00:00+00:00"}],
        ),
    )
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-2",
                "commence_time": "2026-04-04T20:00:00+00:00",
            }
        ]
    )

    assert result == []


def test_log_live_fight_skip_once_dedupes_non_ufc_noise(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monotonic_values = iter([100.0, 101.0])
    monkeypatch.setattr(bot.time, "monotonic", lambda: next(monotonic_values))

    fight = {
        "fighter_a": "Masayuki Kikuiri",
        "fighter_b": "Ernesto Rodriguez",
        "event_id": "evt-1",
        "commence_time": "2026-03-27T23:00:00+00:00",
    }

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(fight, bot._NON_UFC_LIVE_CONTEXT_REASON)
        bot._log_live_fight_skip_once(fight, bot._NON_UFC_LIVE_CONTEXT_REASON)

    skip_records = [
        record for record in caplog.records
        if "Skipping Masayuki Kikuiri vs Ernesto Rodriguez" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO


def test_log_live_fight_skip_once_treats_missing_upcoming_card_skip_as_info(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monkeypatch.setattr(bot.time, "monotonic", lambda: 100.0)

    fight = {
        "fighter_a": "Merab Dvalishvili",
        "fighter_b": "Petr Yan",
        "event_id": "evt-2",
        "commence_time": "2026-06-28T02:00:00+00:00",
    }
    reason = (
        "pair already exists in local UFC history (2025-12-06) "
        "but is not on any upcoming UFC card"
    )

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(fight, reason)

    skip_records = [
        record for record in caplog.records
        if "Skipping Merab Dvalishvili vs Petr Yan" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO


def test_log_live_fight_skip_once_treats_safety_buffer_skip_as_info(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monkeypatch.setattr(bot.time, "monotonic", lambda: 100.0)

    fight = {
        "fighter_a": "Ming Shi",
        "fighter_b": "Puja Tomar",
        "event_id": "evt-3",
        "commence_time": "2026-05-29T13:40:00+00:00",
    }

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(
            fight,
            "fight starts at 2026-05-29T13:40:00+00:00 (safety buffer 1:00:00)",
        )

    skip_records = [
        record for record in caplog.records
        if "Skipping Ming Shi vs Puja Tomar" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO
