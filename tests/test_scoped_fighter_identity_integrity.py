import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import betsapi_mma, fighter_lookup, ufc_active_roster, ufc_refresh
from src.data.name_utils import (
    CURRENT_UFCSTATS_ID_OVERRIDES,
    derive_ambiguous_fighter_name_keys,
    fighter_identity_key,
    normalize_cross_source_name,
)
from src.features import build_features as build_features_module


_REAL_REVERSED_LEGACY_OVERLAYS = [
    pytest.param(
        "2019-12-07",
        "Virna Jandiroba",
        "Mallory Martin",
        ((215.0, -255.0), (None, None), (None, None), (1325.0, 950.0), (1575.0, 275.0), (330.0, -105.0)),
        id="virna-jandiroba-mallory-martin",
    ),
    pytest.param(
        "2020-03-07",
        "Giga Chikadze",
        "Jamall Emmers",
        ((-170.0, 160.0), (None, None), (None, None), (450.0, 400.0), (275.0, 1700.0), (240.0, 285.0)),
        id="giga-chikadze-jamall-emmers",
    ),
    pytest.param(
        "2020-05-16",
        "Anthony Hernandez",
        "Kevin Holland",
        ((-110.0, -110.0), (None, None), (None, None), (686.0, 1405.0), (540.0, 385.0), (200.0, 230.0)),
        id="anthony-hernandez-kevin-holland",
    ),
    pytest.param(
        "2020-05-16",
        "Darren Elkins",
        "Nate Landwehr",
        ((100.0, -120.0), (None, None), (None, None), (310.0, 950.0), (1090.0, 614.0), (320.0, 150.0)),
        id="darren-elkins-nate-landwehr",
    ),
    pytest.param(
        "2020-05-16",
        "Eryk Anders",
        "Krzysztof Jotko",
        ((-105.0, -105.0), (None, None), (None, None), (800.0, 285.0), (1025.0, 3550.0), (125.0, 330.0)),
        id="eryk-anders-krzysztof-jotko",
    ),
    pytest.param(
        "2020-05-16",
        "Matt Brown",
        "Miguel Baeza",
        ((-155.0, -155.0), (None, None), (None, None), (170.0, 245.0), (850.0, 1350.0), (430.0, 450.0)),
        id="matt-brown-miguel-baeza",
    ),
]


def test_collision_identity_keys_require_ids_and_keep_namesakes_separate():
    collision_keys = derive_ambiguous_fighter_name_keys(
        [
            {"name": "Future Namesake", "fighter_id": "1111111111111111"},
            {"name": "Future Namesake", "fighter_id": "2222222222222222"},
        ]
    )

    assert fighter_identity_key(
        "Future Namesake", ambiguous_name_keys=collision_keys
    ) is None
    assert fighter_identity_key(
        "Future Namesake",
        "1111111111111111",
        ambiguous_name_keys=collision_keys,
    ) == "ufcstats:1111111111111111"
    assert fighter_identity_key("Jean Silva") is None
    assert fighter_identity_key("Jean Silva", "9211aae062b799d6") != fighter_identity_key(
        "Jean Silva", "52ef95b5860fb28c"
    )


def test_name_only_live_lookup_fails_closed_for_collision(monkeypatch):
    monkeypatch.setattr(
        fighter_lookup,
        "_get_soup",
        lambda _url: pytest.fail("ambiguous name-only lookup must not make a request"),
    )

    assert fighter_lookup.search_fighter_url("Jean Silva") is None
    assert fighter_lookup.lookup_fighter("Jean Silva") is None
    assert fighter_lookup.search_fighter_url(
        "Jean Silva", fighter_id="52ef95b5860fb28c"
    ) == "http://ufcstats.com/fighter-details/52ef95b5860fb28c"


def test_live_feature_contract_preserves_ids_and_nullable_context(monkeypatch):
    def fake_lookup(name, **_kwargs):
        return {"profile": {"name": name}, "features": {}, "fights": []}

    monkeypatch.setattr(fighter_lookup, "_call_lookup_fighter", fake_lookup)
    spec = SimpleNamespace(
        name="identity_context_test",
        feature_cols=["is_title_bout", "is_empty_arena", "num_rounds_feat"],
    )

    features = fighter_lookup.build_fight_features(
        "Jean Silva",
        "Opponent",
        fighter_a_id="52ef95b5860fb28c",
        is_title_bout=None,
        is_empty_arena=None,
        num_rounds=None,
        training_spec=spec,
    )

    assert features["fighter_a_id"] == "52ef95b5860fb28c"
    assert features["fighter_b_id"] is None
    assert np.isnan(features["is_title_bout"])
    assert np.isnan(features["is_empty_arena"])
    assert np.isnan(features["num_rounds_feat"])


def test_unavailable_h2h_and_debut_weight_move_stay_unknown():
    is_rematch, h2h_diff = fighter_lookup._h2h_summary("A", "B", None, None)
    assert np.isnan(is_rematch)
    assert np.isnan(h2h_diff)
    assert np.isnan(
        fighter_lookup._compute_wc_move_from_history(
            {"fights": []}, "Lightweight"
        )
    )


def test_training_histories_separate_old_and_modern_jean_silva(monkeypatch):
    monkeypatch.setattr(
        build_features_module,
        "_load_pre_ufc_supplement",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        build_features_module,
        "_load_amateur_supplement",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    fights = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2005-01-01"),
                "fighter_a": "Jean Silva",
                "fighter_a_id": "9211aae062b799d6",
                "fighter_b": "Old Opponent",
                "winner": "Jean Silva",
                "weight_class": "Lightweight",
            },
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Jean Silva",
                "fighter_a_id": "52ef95b5860fb28c",
                "fighter_b": "Modern Opponent One",
                "winner": "Jean Silva",
                "weight_class": "Featherweight",
            },
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "Jean Silva",
                "fighter_a_id": "52ef95b5860fb28c",
                "fighter_b": "Modern Opponent Two",
                "winner": "Jean Silva",
                "weight_class": "Featherweight",
            },
        ]
    )

    features = build_features_module.build_features(fights)
    modern_second = features.loc[features["event_date"] == pd.Timestamp("2025-01-01")].iloc[0]

    assert modern_second["fighter_a_id"] == "52ef95b5860fb28c"
    assert modern_second["a_num_fights"] == pytest.approx(1.0)
    assert modern_second["a_wc_move"] == pytest.approx(0.0)


def test_ambiguous_training_row_without_id_blocks_artifact_build():
    fights = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "Bruno Silva",
                "fighter_b": "Opponent",
                "winner": "Bruno Silva",
            }
        ]
    )

    with pytest.raises(build_features_module.UnresolvedTrainingFighterIdentityError):
        build_features_module.build_features(fights)


def test_missing_pulled_stats_are_nan_but_observed_zero_is_preserved():
    missing = ufc_refresh._parse_landed_attempted("--")
    observed_zero = ufc_refresh._parse_landed_attempted("0 of 0")

    assert all(np.isnan(value) for value in missing)
    assert observed_zero == (0.0, 0.0)
    assert np.isnan(ufc_refresh._safe_number_or_nan("--"))
    assert ufc_refresh._safe_number_or_nan("0") == 0.0


def test_active_roster_reviewed_current_ids_override_stale_name_candidate():
    stale = {
        "name": "Jean Silva",
        "fighter_url": "http://ufcstats.com/fighter-details/9211aae062b799d6",
        "source": "fixture",
    }
    candidates = {normalize_cross_source_name("Jean Silva"): [stale]}

    resolved = ufc_active_roster._resolve_local_ufcstats_profile(
        {"official_name": "Jean Silva"}, candidates=candidates
    )

    assert resolved["ufcstats_url"].endswith(
        CURRENT_UFCSTATS_ID_OVERRIDES["jean silva"]
    )
    assert resolved["ufcstats_resolution"] == "reviewed_current_identity"


def test_committed_active_roster_has_current_collision_ids():
    roster_path = Path("data/raw/ufc_active_roster_official.csv")
    roster = pd.read_csv(roster_path).set_index("official_name")

    for fighter in ("Jean Silva", "Mike Davis", "Victor Valenzuela"):
        expected_id = CURRENT_UFCSTATS_ID_OVERRIDES[fighter.casefold()]
        assert str(roster.loc[fighter, "ufcstats_url"]).endswith(expected_id)
        assert roster.loc[fighter, "ufcstats_resolution"] == "reviewed_current_identity"


def test_reversed_legacy_overlay_swaps_every_side_pair(monkeypatch):
    for helper_name in (
        "_historical_moneyline_overlay",
        "_historical_opening_moneyline_overlay",
        "_historical_rankings_overlay",
        "_historical_method_odds_overlay",
    ):
        monkeypatch.setattr(ufc_refresh, helper_name, lambda *_args, **_kwargs: pd.DataFrame())

    roots = ("odds", "wc_rank", "pfp_rank", "ko_odds", "sub_odds", "dec_odds")
    pulled = pd.DataFrame(
        [{"event_date": "2025-01-01", "fighter_a": "Alpha", "fighter_b": "Beta"}]
    )
    legacy_row = {
        "event_date": "2025-01-01",
        "fighter_a": "Beta",
        "fighter_b": "Alpha",
    }
    for index, root in enumerate(roots, start=1):
        legacy_row[f"a_{root}"] = float(index * 10 + 1)
        legacy_row[f"b_{root}"] = float(index * 10 + 2)

    overlaid = ufc_refresh.build_pulled_all_plus_legacy_market_training_dataset(
        pd.DataFrame([legacy_row]), pulled
    ).iloc[0]

    for index, root in enumerate(roots, start=1):
        assert overlaid[f"a_{root}"] == pytest.approx(float(index * 10 + 2))
        assert overlaid[f"b_{root}"] == pytest.approx(float(index * 10 + 1))


@pytest.mark.parametrize(
    ("event_date", "pulled_a", "pulled_b", "source_pairs"),
    _REAL_REVERSED_LEGACY_OVERLAYS,
)
def test_all_six_reviewed_real_legacy_reversals_orient_every_pair(
    monkeypatch,
    event_date,
    pulled_a,
    pulled_b,
    source_pairs,
):
    """Pin the six reversed joins observed in the current raw datasets."""
    for helper_name in (
        "_historical_moneyline_overlay",
        "_historical_opening_moneyline_overlay",
        "_historical_rankings_overlay",
        "_historical_method_odds_overlay",
    ):
        monkeypatch.setattr(ufc_refresh, helper_name, lambda *_args, **_kwargs: pd.DataFrame())

    roots = ("odds", "wc_rank", "pfp_rank", "ko_odds", "sub_odds", "dec_odds")
    pulled = pd.DataFrame(
        [{"event_date": event_date, "fighter_a": pulled_a, "fighter_b": pulled_b}]
    )
    legacy_row = {
        "event_date": event_date,
        "fighter_a": pulled_b,
        "fighter_b": pulled_a,
    }
    for root, (source_a, source_b) in zip(roots, source_pairs):
        legacy_row[f"a_{root}"] = source_a
        legacy_row[f"b_{root}"] = source_b

    oriented = ufc_refresh.build_pulled_all_plus_legacy_market_training_dataset(
        pd.DataFrame([legacy_row]), pulled
    ).iloc[0]

    for root, (source_a, source_b) in zip(roots, source_pairs):
        if source_b is None:
            assert pd.isna(oriented[f"a_{root}"])
        else:
            assert oriented[f"a_{root}"] == pytest.approx(source_b)
        if source_a is None:
            assert pd.isna(oriented[f"b_{root}"])
        else:
            assert oriented[f"b_{root}"] == pytest.approx(source_a)


def test_betsapi_reversed_and_unmatched_rows_are_oriented_or_rejected():
    fight_key = betsapi_mma._fight_key("Alpha", "Beta", "2025-01-01")
    left = pd.DataFrame(
        [{"fight_key": fight_key, "fighter_a": "Beta", "fighter_b": "Alpha"}]
    )
    reversed_source = pd.DataFrame(
        [
            {
                "fight_key": fight_key,
                betsapi_mma._BETSAPI_SOURCE_FIGHTER_A: "Alpha",
                betsapi_mma._BETSAPI_SOURCE_FIGHTER_B: "Beta",
                "betsapi_a_implied_prob": 0.7,
                "betsapi_b_implied_prob": 0.3,
                "betsapi_diff_implied_prob": 0.4,
                "betsapi_hist_handicap_direction": -1.5,
            }
        ]
    )

    oriented = betsapi_mma._merge_betsapi_with_orientation(
        left, reversed_source
    ).iloc[0]
    assert oriented["betsapi_a_implied_prob"] == pytest.approx(0.3)
    assert oriented["betsapi_b_implied_prob"] == pytest.approx(0.7)
    assert oriented["betsapi_diff_implied_prob"] == pytest.approx(-0.4)
    assert oriented["betsapi_hist_handicap_direction"] == pytest.approx(1.5)

    unmatched_source = reversed_source.copy()
    unmatched_source[betsapi_mma._BETSAPI_SOURCE_FIGHTER_A] = "Other One"
    unmatched_source[betsapi_mma._BETSAPI_SOURCE_FIGHTER_B] = "Other Two"
    rejected = betsapi_mma._merge_betsapi_with_orientation(
        left, unmatched_source
    ).iloc[0]
    assert np.isnan(rejected["betsapi_a_implied_prob"])
    assert np.isnan(rejected["betsapi_diff_implied_prob"])


def test_live_cycle_skips_only_ambiguous_identity_bout(monkeypatch, tmp_path):
    """One ambiguous live lookup must not abort prediction of the next bout."""
    from datetime import datetime, timezone

    from src import bot

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    artifact_path = tmp_path / "xgboost_model.pkl"
    artifact_path.write_text("fixture", encoding="utf-8")
    model_result = {
        "feature_cols": [],
        "col_medians": np.array([]),
        "feature_importance": {},
        "raw_model": None,
        "artifact_path": str(artifact_path),
        "training_spec": {"name": "identity-live-test", "feature_cols": []},
    }
    fights = [
        {
            "event_id": "ambiguous-bout",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Jean Silva",
            "fighter_b": "First Opponent",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        },
        {
            "event_id": "valid-bout",
            "commence_time": "2026-03-28T21:00:00Z",
            "fighter_a": "Valid Fighter",
            "fighter_b": "Second Opponent",
            "a_fair_prob_avg": 0.52,
            "b_fair_prob_avg": 0.48,
            "num_bookmakers": 8,
        },
    ]

    class FakeOddsClient:
        def get_live_odds(self):
            return []

        def odds_to_dataframe(self, _odds):
            return pd.DataFrame()

        def get_consensus_odds(self, _odds_df):
            return pd.DataFrame(fights)

    build_calls = []
    prediction_calls = []
    trader_predictions = []

    def fake_build_fight_features(fighter_a, fighter_b, **_kwargs):
        build_calls.append((fighter_a, fighter_b))
        if fighter_a == "Jean Silva":
            raise fighter_lookup.AmbiguousFighterIdentityError("missing reviewed UFCStats ID")
        return {
            "a_num_fights": 1,
            "b_num_fights": 1,
        }, {
            "fighter_a_source": "ufcstats",
            "fighter_a_fight_history_status": "complete",
            "fighter_b_source": "ufcstats",
            "fighter_b_fight_history_status": "complete",
        }

    def fake_predict_fight(_features, **_kwargs):
        prediction_calls.append("called")
        return {"prob_a": 0.58, "prob_b": 0.42, "confidence": 0.58}

    def fake_run_duo_traders(**kwargs):
        trader_predictions.extend(kwargs["predictions"].to_dict(orient="records"))
        return {"total_orders": 0}

    monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc),
    )
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
    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
    monkeypatch.setattr("src.model.predict.predict_fight", fake_predict_fight)
    monkeypatch.setattr(
        "src.data.fighter_lookup.build_fight_features", fake_build_fight_features
    )
    monkeypatch.setattr(
        "src.data.line_tracker.detect_injury_or_cancellation",
        lambda *_args, **_kwargs: {"suspected": False},
    )
    monkeypatch.setattr(
        "src.polymarket.markets.get_ufc_fight_markets",
        lambda **_kwargs: pd.DataFrame([{"slug": "ufc-card"}]),
    )
    monkeypatch.setattr(
        "src.strategy.duo_trader.run_duo_traders", fake_run_duo_traders
    )

    result = bot.cmd_duo_live(
        type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})()
    )

    assert build_calls == [
        ("Jean Silva", "First Opponent"),
        ("Valid Fighter", "Second Opponent"),
    ]
    assert prediction_calls == ["called"]
    assert [row["fighter_a"] for row in trader_predictions] == ["Valid Fighter"]
    assert result["total_orders"] == 0
    cache = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
    assert [row["fighter_a"] for row in cache["predictions"]] == ["Valid Fighter"]
