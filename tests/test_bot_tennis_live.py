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


def test_assert_tennis_real_trading_allowed_raises_concrete_message(monkeypatch):
    monkeypatch.delenv(bot.TENNIS_REAL_TRADING_ARM_ENV, raising=False)
    monkeypatch.delenv(bot.TENNIS_REAL_TRADING_CONFIRM_ENV, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        bot.assert_tennis_real_trading_allowed(source="test")

    message = str(excinfo.value)
    assert f"{bot.TENNIS_REAL_TRADING_ARM_ENV}=1" in message
    assert (
        f"{bot.TENNIS_REAL_TRADING_CONFIRM_ENV}={bot.TENNIS_REAL_TRADING_CONFIRM_VALUE}"
        in message
    )


def test_cmd_tennis_live_non_dry_run_requires_tennis_trading_auth(monkeypatch):
    monkeypatch.delenv(bot.TENNIS_REAL_TRADING_ARM_ENV, raising=False)
    monkeypatch.delenv(bot.TENNIS_REAL_TRADING_CONFIRM_ENV, raising=False)

    errors = []
    monkeypatch.setattr(bot.logger, "error", lambda msg, *a: errors.append(msg % a if a else msg))
    monkeypatch.setattr(
        bot,
        "_build_tennis_prediction_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected prediction build")),
    )

    result = bot.cmd_tennis_live(argparse.Namespace(dry_run=False, model="surface_elo", min_edge=None))

    assert result["status"] == "error"
    assert bot.TENNIS_REAL_TRADING_ARM_ENV in result["reason"]
    assert len(errors) == 1


def test_cmd_tennis_live_non_dry_run_proceeds_when_tennis_trading_is_armed(monkeypatch):
    warnings = []
    monkeypatch.setenv(bot.TENNIS_REAL_TRADING_ARM_ENV, "1")
    monkeypatch.setenv(bot.TENNIS_REAL_TRADING_CONFIRM_ENV, bot.TENNIS_REAL_TRADING_CONFIRM_VALUE)
    monkeypatch.setattr(bot.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg))
    monkeypatch.setattr(
        bot,
        "_build_tennis_prediction_frame",
        lambda *args, **kwargs: None,  # returns None → early exit
    )

    bot.cmd_tennis_live(argparse.Namespace(dry_run=False, model="surface_elo", min_edge=None))
    assert any("non-dry-run mode enabled" in w.lower() for w in warnings)


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


def test_build_tennis_prediction_frame_rebuilds_missing_runtime_model(monkeypatch):
    import src.data.tennis_data as tennis_data
    import src.data.tennis_odds as tennis_odds
    import src.features.tennis_features as tennis_features
    import src.model.tennis_model as tennis_model

    history_df = pd.DataFrame(
        [
            {"event_date": pd.Timestamp("2026-03-20"), "tour": "atp", "tourney_name": "Miami", "match_num": 1},
        ]
    )
    training_features = pd.DataFrame(
        [
            {"event_date": pd.Timestamp("2026-03-20"), "target": 1, "a_num_matches": 8, "b_num_matches": 9},
        ]
    )

    monkeypatch.setattr(tennis_data, "load_processed_tennis_data", lambda: history_df)
    monkeypatch.setattr(
        tennis_odds,
        "fetch_live_tennis_consensus",
        lambda: pd.DataFrame([{"fighter_a": "Regular A", "fighter_b": "Regular B", "tour": "atp"}]),
    )
    monkeypatch.setattr(
        tennis_model,
        "load_tennis_model",
        lambda model_name="lean_hybrid": (_ for _ in ()).throw(FileNotFoundError("missing model")),
    )

    captured = {}

    def fake_build_training_features(matches_df):
        captured["history_rows"] = len(matches_df)
        return training_features

    def fake_ensure_model(model_name="lean_hybrid", *, features_df=None, min_matches=3):
        captured["model_name"] = model_name
        captured["feature_rows"] = len(features_df)
        captured["min_matches"] = min_matches
        return {"model": object()}

    monkeypatch.setattr(tennis_features, "build_tennis_features", fake_build_training_features)
    monkeypatch.setattr(tennis_model, "ensure_tennis_model", fake_ensure_model)
    monkeypatch.setattr(
        tennis_features,
        "build_live_tennis_features",
        lambda matchups_df, history_df: pd.DataFrame(
            [{"fighter_a": "Regular A", "fighter_b": "Regular B", "a_num_matches": 7, "b_num_matches": 9}]
        ),
    )
    monkeypatch.setattr(
        tennis_model,
        "predict_tennis_batch",
        lambda frame, model_result=None: frame.assign(prob_a=0.58, prob_b=0.42),
    )

    predictions = bot._build_tennis_prediction_frame(model_name="lean_hybrid")

    assert len(predictions) == 1
    assert captured == {
        "history_rows": 1,
        "model_name": "lean_hybrid",
        "feature_rows": 1,
        "min_matches": 3,
    }


def test_build_tennis_prediction_frame_enriches_live_rankings_before_feature_build(monkeypatch):
    import src.data.tennis_data as tennis_data
    import src.data.tennis_odds as tennis_odds
    import src.data.tennis_rankings_history as tennis_rankings_history
    import src.features.tennis_features as tennis_features
    import src.model.tennis_model as tennis_model

    history_df = pd.DataFrame(
        [
            {"event_date": pd.Timestamp("2026-03-20"), "tour": "atp", "tourney_name": "Miami", "match_num": 1},
        ]
    )
    consensus_df = pd.DataFrame(
        [
            {
                "fighter_a": "Frances Tiafoe",
                "fighter_b": "Jakub Mensik",
                "tour": "atp",
                "commence_time": "2026-03-23T18:00:00Z",
            }
        ]
    )

    monkeypatch.setattr(tennis_data, "load_processed_tennis_data", lambda: history_df)
    monkeypatch.setattr(tennis_odds, "fetch_live_tennis_consensus", lambda: consensus_df.copy())
    monkeypatch.setattr(tennis_model, "load_tennis_model", lambda model_name="lean_hybrid": {"model": object()})
    monkeypatch.setattr(
        tennis_rankings_history,
        "enrich_live_tennis_matchups_with_current_rankings",
        lambda frame, **kwargs: frame.assign(
            player_a_rank=17.0,
            player_b_rank=12.0,
            player_a_rank_points=2585.0,
            player_b_rank_points=4290.0,
        ),
    )

    captured = {}

    def fake_build_live_tennis_features(matchups_df, history_df):
        captured["live_input"] = matchups_df[
            ["fighter_a", "fighter_b", "player_a_rank", "player_b_rank", "player_b_rank_points"]
        ].to_dict("records")
        return pd.DataFrame(
            [
                {
                    "fighter_a": "Frances Tiafoe",
                    "fighter_b": "Jakub Mensik",
                    "a_num_matches": 7,
                    "b_num_matches": 9,
                }
            ]
        )

    monkeypatch.setattr(tennis_features, "build_live_tennis_features", fake_build_live_tennis_features)
    monkeypatch.setattr(
        tennis_model,
        "predict_tennis_batch",
        lambda frame, model_result=None: frame.assign(prob_a=0.58, prob_b=0.42),
    )

    predictions = bot._build_tennis_prediction_frame(model_name="lean_hybrid")

    assert len(predictions) == 1
    assert captured["live_input"] == [
        {
            "fighter_a": "Frances Tiafoe",
            "fighter_b": "Jakub Mensik",
            "player_a_rank": 17.0,
            "player_b_rank": 12.0,
            "player_b_rank_points": 4290.0,
        }
    ]


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
