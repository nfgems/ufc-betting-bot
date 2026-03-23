import argparse
import sys

import pandas as pd
import pytest

import src.bot as bot


def test_tennis_live_parser_defaults_to_dry_run(monkeypatch):
    captured = {}

    def _capture(args):
        captured["dry_run"] = args.dry_run

    monkeypatch.setattr(bot, "cmd_tennis_live", _capture)
    monkeypatch.setattr(sys, "argv", ["bot", "tennis-live"])

    bot.main()

    assert captured["dry_run"] is True


def test_tennis_live_parser_allows_guard_to_receive_no_dry_run(monkeypatch):
    captured = {}

    def _capture(args):
        captured["dry_run"] = args.dry_run

    monkeypatch.setattr(bot, "cmd_tennis_live", _capture)
    monkeypatch.setattr(sys, "argv", ["bot", "tennis-live", "--no-dry-run"])

    bot.main()

    assert captured["dry_run"] is False


def test_cmd_tennis_live_non_dry_run_proceeds_to_predictions(monkeypatch):
    """After removing the TENNIS_LIVE_TRADING_ARMED gate, --no-dry-run proceeds normally."""
    warnings = []
    monkeypatch.setattr(bot.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg))
    monkeypatch.setattr(
        bot,
        "_build_tennis_prediction_frame",
        lambda *args, **kwargs: None,  # returns None → early exit
    )

    bot.cmd_tennis_live(argparse.Namespace(dry_run=False, model="surface_elo", min_edge=None))
    assert any("live trading enabled" in w.lower() for w in warnings)


def test_build_tennis_prediction_frame_applies_minimum_history_gate(monkeypatch):
    import src.data.tennis_data as tennis_data
    import src.data.tennis_odds as tennis_odds
    import src.features.tennis_features as tennis_features
    import src.model.tennis_model as tennis_model

    monkeypatch.setattr(
        tennis_data,
        "load_processed_tennis_data",
        lambda: pd.DataFrame([{"event_date": pd.Timestamp("2026-03-20")}]),
    )
    monkeypatch.setattr(
        tennis_odds,
        "fetch_live_tennis_consensus",
        lambda: pd.DataFrame(
            [
                {"fighter_a": "New Player", "fighter_b": "Veteran", "tour": "wta"},
                {"fighter_a": "Regular A", "fighter_b": "Regular B", "tour": "atp"},
            ]
        ),
    )
    monkeypatch.setattr(tennis_model, "load_tennis_model", lambda model_name="surface_elo": {"model": object()})
    monkeypatch.setattr(
        tennis_features,
        "build_live_tennis_features",
        lambda matchups_df, history_df: pd.DataFrame(
            [
                {"fighter_a": "New Player", "fighter_b": "Veteran", "a_num_matches": 1, "b_num_matches": 10},
                {"fighter_a": "Regular A", "fighter_b": "Regular B", "a_num_matches": 7, "b_num_matches": 9},
            ]
        ),
    )

    captured = {}

    def fake_predict(frame, model_result=None):
        captured["fighters"] = frame[["fighter_a", "fighter_b"]].to_dict("records")
        return frame.assign(prob_a=0.6, prob_b=0.4)

    monkeypatch.setattr(tennis_model, "predict_tennis_batch", fake_predict)

    predictions = bot._build_tennis_prediction_frame()

    assert len(predictions) == 1
    assert captured["fighters"] == [{"fighter_a": "Regular A", "fighter_b": "Regular B"}]


def test_cmd_tennis_refresh_daily_forces_current_year_refresh(monkeypatch):
    import src.data.tennis_data as tennis_data
    import src.data.tennis_odds as tennis_odds

    captured = {}

    def fake_prepare_tennis_data(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"event_date": pd.Timestamp("2026-03-22")}])

    monkeypatch.setattr(tennis_data, "prepare_tennis_data", fake_prepare_tennis_data)
    monkeypatch.setattr(tennis_odds, "fetch_live_tennis_consensus", lambda: pd.DataFrame())

    args = argparse.Namespace(
        start_year=2022,
        end_year=2026,
        force_download=False,
        skip_live_seeds=False,
    )

    bot.cmd_tennis_refresh_daily(args)

    assert captured["refresh_current_year"] is True
