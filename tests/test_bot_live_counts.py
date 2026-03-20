import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src import bot


def _make_repo_local_tmp_dir() -> Path:
    path = Path.cwd() / "data" / f"bot-live-context-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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
                    "fighter_a,fighter_b,weight_class,event_date",
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
                    "fighter_a,fighter_b,weight_class,event_date",
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


def test_cmd_duo_live_caches_predictions_when_context_falls_back_to_raw_history(
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
                    "fighter_a,fighter_b,weight_class,event_date",
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
