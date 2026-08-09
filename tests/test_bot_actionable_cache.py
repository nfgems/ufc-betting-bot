import copy
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import bot


NOW = datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc)


def _fight(**updates):
    row = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Ricky Simon",
        "fighter_b": "Adrian Yanez",
        "a_fair_prob_avg": 0.55,
        "b_fair_prob_avg": 0.45,
        "num_bookmakers": 8,
    }
    row.update(updates)
    return row


def _event_context():
    return {
        "event_id": "evt-1",
        "event_date": "2026-03-28",
        "weight_class": "Bantamweight",
        "num_rounds": 3,
        "is_title_bout": False,
        "is_empty_arena": False,
    }


def _model_result(artifact_path):
    return {
        "feature_cols": ["a_num_fights"],
        "col_medians": np.array([0.0]),
        "feature_importance": {},
        "raw_model": None,
        "artifact_path": str(artifact_path),
        "training_spec": {
            "name": "actionable-cache-test",
            "feature_cols": ["a_num_fights"],
        },
    }


def _cached_row(fight, runtime_signature, *, generated_at=None):
    context_snapshot = bot._prediction_event_context_snapshot(fight, _event_context())
    quality = {"blocked": False, "reasons": []}
    return {
        "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
        "fighter_a": fight["fighter_a"],
        "fighter_b": fight["fighter_b"],
        "prob_a": 0.61,
        "prob_b": 0.39,
        "confidence": 0.61,
        "event_id": fight["event_id"],
        "event_date": fight["commence_time"],
        "card_date": "2026-03-28",
        "a_market_prob": fight["a_fair_prob_avg"],
        "b_market_prob": fight["b_fair_prob_avg"],
        "no_odds_prob_a": None,
        "no_odds_prob_b": None,
        "a_num_fights": 5,
        "b_num_fights": 6,
        "trade_blocked": False,
        "trade_block_reason": "",
        "data_quality": quality,
        "pair_key": bot._live_fight_pair_key(fight["fighter_a"], fight["fighter_b"]),
        "cache_key": bot._prediction_cache_key(fight),
        "prediction_generated_at": (generated_at or (NOW - timedelta(minutes=10))).isoformat(),
        "odds_snapshot": bot._prediction_odds_snapshot(fight),
        "prediction_input_odds_snapshot": bot._prediction_odds_snapshot(fight),
        "prediction_input_line_features": {},
        "method_odds_fingerprint": "method-odds:not-requested",
        "event_context_snapshot": context_snapshot,
        "runtime_signature": runtime_signature,
        "model_features": {"a_num_fights": 5},
        "feature_provenance": {"data_quality": quality},
    }


def _actionable(row, fight, runtime_signature, *, context_snapshot=None):
    return bot._actionable_cached_prediction(
        row,
        fight,
        runtime_signature=runtime_signature,
        inference_spec=SimpleNamespace(feature_cols=["a_num_fights"]),
        current_event_context_snapshot=(
            context_snapshot
            if context_snapshot is not None
            else bot._prediction_event_context_snapshot(fight, _event_context())
        ),
        require_no_odds_prediction=False,
        now=NOW,
    )


def test_actionable_cache_rejects_stale_changed_and_blocked_rows(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    runtime_signature = bot._prediction_runtime_signature(
        model_result=_model_result(artifact),
        no_odds_result=None,
        runtime_bundle_summary=None,
    )
    fight = _fight()
    base = _cached_row(fight, runtime_signature)

    stale = copy.deepcopy(base)
    stale["prediction_generated_at"] = (NOW - timedelta(minutes=20, seconds=1)).isoformat()
    assert _actionable(stale, fight, runtime_signature)[0] is None

    changed_fight = _fight(a_fair_prob_avg=0.581, b_fair_prob_avg=0.419)
    assert _actionable(base, changed_fight, runtime_signature)[0] is None

    changed_context = bot._prediction_event_context_snapshot(
        fight,
        {**_event_context(), "num_rounds": 5},
    )
    assert _actionable(base, fight, runtime_signature, context_snapshot=changed_context)[0] is None

    blocked = copy.deepcopy(base)
    blocked["trade_blocked"] = True
    blocked["data_quality"]["blocked"] = True
    assert _actionable(blocked, fight, runtime_signature)[0] is None


def test_cmd_duo_live_executes_eligible_cache_once_before_injury_or_line_checks(
    monkeypatch,
    tmp_path,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    model_result = _model_result(artifact)
    runtime_signature = bot._prediction_runtime_signature(
        model_result=model_result,
        no_odds_result=None,
        runtime_bundle_summary=None,
    )
    fight = _fight()
    cached_row = _cached_row(fight, runtime_signature)
    (logs_dir / "predictions_cache.json").write_text(
        json.dumps(
            {
                "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                "timestamp": (NOW - timedelta(minutes=10)).isoformat(),
                "refresh_in_progress": False,
                "predictions": [cached_row],
                "global_feature_importance": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeOddsClient:
        def get_live_odds(self):
            return []

        def odds_to_dataframe(self, _raw):
            return pd.DataFrame()

        def get_consensus_odds(self, _frame):
            return pd.DataFrame([fight])

    events = []
    runner_calls = []
    tracker_flags = []

    def fake_runner(**kwargs):
        events.append("runner")
        runner_calls.append(kwargs["predictions"].copy())
        tracker_flags.append(kwargs.get("run_model_tracker", True))
        return {"total_orders": 2}

    def fake_injury(*_args, **_kwargs):
        assert events and events[0] == "runner"
        events.append("injury")
        return {"suspected": False}

    def fake_line(*_args, **_kwargs):
        assert events and events[0] == "runner"
        events.append("line")
        return {}

    monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(bot, "_current_utc", lambda: NOW)
    monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_resolve_runtime_bundle_summary", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "_runtime_completed_ufc_event_dates_before", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_load_live_event_contexts_for_fights", lambda *_args, **_kwargs: [_event_context()])
    monkeypatch.setattr(bot, "_resolve_live_event_context", lambda *_args, **_kwargs: _event_context())
    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", fake_injury)
    monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", fake_line)
    monkeypatch.setattr(
        "src.polymarket.markets.get_ufc_fight_markets",
        lambda **_kwargs: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
    )
    monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", fake_runner)

    result = bot.cmd_duo_live(
        SimpleNamespace(model="xgboost", dry_run=True, min_edge=0.02)
    )

    assert result == {"status": "ok", "total_orders": 4}
    assert events[0] == "runner"
    assert events.count("runner") == 2
    assert len(runner_calls) == 2
    assert runner_calls[0]["fighter_a"].tolist() == ["Ricky Simon"]
    assert runner_calls[1]["fighter_a"].tolist() == ["Ricky Simon"]
    assert tracker_flags == [False, True]

    # If injury blocking is armed, the fast lane must preserve that gate even
    # though its normal purpose is to avoid slow enrichment work.
    events.clear()
    runner_calls.clear()
    tracker_flags.clear()
    monkeypatch.setattr("src.config.INJURY_BLOCK_BETS", True)
    monkeypatch.setattr(
        "src.data.line_tracker.detect_injury_or_cancellation",
        lambda *_args, **_kwargs: {
            "suspected": True,
            "severity": "block",
            "reason": "fight cancelled",
        },
    )

    blocked_result = bot.cmd_duo_live(
        SimpleNamespace(model="xgboost", dry_run=True, min_edge=0.02)
    )

    assert blocked_result["status"] == "idle"
    assert runner_calls == []


def test_cmd_duo_live_executes_uncached_build_after_early_cached_portfolio(
    monkeypatch,
    tmp_path,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    model_result = _model_result(artifact)
    runtime_signature = bot._prediction_runtime_signature(
        model_result=model_result,
        no_odds_result=None,
        runtime_bundle_summary=None,
    )
    cached_fight = _fight()
    uncached_fight = _fight(
        event_id="evt-2",
        commence_time="2026-03-28T21:00:00Z",
        fighter_a="Fresh Fighter",
        fighter_b="New Opponent",
        a_fair_prob_avg=0.52,
        b_fair_prob_avg=0.48,
    )
    (logs_dir / "predictions_cache.json").write_text(
        json.dumps(
            {
                "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                "timestamp": (NOW - timedelta(minutes=10)).isoformat(),
                "refresh_in_progress": False,
                "predictions": [_cached_row(cached_fight, runtime_signature)],
                "global_feature_importance": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeOddsClient:
        def get_live_odds(self):
            return []

        def odds_to_dataframe(self, _raw):
            return pd.DataFrame()

        def get_consensus_odds(self, _frame):
            return pd.DataFrame([cached_fight, uncached_fight])

    runner_calls = []
    tracker_flags = []

    def fake_runner(**kwargs):
        runner_calls.append(kwargs["predictions"].copy())
        tracker_flags.append(kwargs.get("run_model_tracker", True))
        return {"total_orders": len(runner_calls) + 1}

    monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(bot, "_current_utc", lambda: NOW)
    monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_resolve_runtime_bundle_summary", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "_runtime_completed_ufc_event_dates_before", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_load_live_event_contexts_for_fights", lambda *_args, **_kwargs: [_event_context()])
    monkeypatch.setattr(bot, "_resolve_live_event_context", lambda *_args, **_kwargs: _event_context())
    monkeypatch.setattr(
        bot,
        "_live_prediction_quality_assessment",
        lambda *_args, **_kwargs: {"blocked": False, "reasons": []},
    )
    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
    monkeypatch.setattr(
        "src.model.predict.predict_fight",
        lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6},
    )
    monkeypatch.setattr(
        "src.data.fighter_lookup.build_fight_features",
        lambda *_args, **_kwargs: (
            {"a_num_fights": 5, "b_num_fights": 5},
            {"fighter_a_source": "ufcstats", "fighter_b_source": "ufcstats"},
        ),
    )
    monkeypatch.setattr(
        "src.data.line_tracker.detect_injury_or_cancellation",
        lambda *_args, **_kwargs: {"suspected": False},
    )
    monkeypatch.setattr(
        "src.data.line_tracker.get_line_movement_features",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "src.polymarket.markets.get_ufc_fight_markets",
        lambda **_kwargs: pd.DataFrame([{"slug": "ufc-test-market"}]),
    )
    monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", fake_runner)

    result = bot.cmd_duo_live(
        SimpleNamespace(model="xgboost", dry_run=True, min_edge=0.02)
    )

    assert result == {"status": "ok", "total_orders": 5}
    assert len(runner_calls) == 2
    assert runner_calls[0]["fighter_a"].tolist() == ["Ricky Simon"]
    assert runner_calls[1]["fighter_a"].tolist() == ["Ricky Simon", "Fresh Fighter"]
    assert tracker_flags == [False, True]
