from __future__ import annotations

import pytest
import pandas as pd

from src import bot
from src.data import live_monitor, ufc_active_roster
from src.model.production_bundle import (
    ProductionBundleError,
    _registered_contract_payload,
)
from src.model.training_spec import (
    V4_METHOD_ODDS_FEATURE_COLS,
    full_live_contract_v6_integrity_205_fullfit_spec,
    full_live_contract_v6_integrity_205_spec,
    full_live_contract_v6_integrity_211_fullfit_spec,
    full_live_contract_v6_integrity_211_spec,
)


def test_legacy_deferred_odds_seed_normalizes_to_explicit_booster_seed():
    deferred = {
        "name": "legacy",
        "odds_noise_seed": None,
        "xgb_params": {"random_state": 42},
    }
    explicit = {
        "name": "legacy",
        "odds_noise_seed": 42,
        "xgb_params": {"random_state": 42},
    }

    assert _registered_contract_payload(deferred) == _registered_contract_payload(
        explicit
    )


def test_deferred_odds_seed_without_booster_seed_fails_closed():
    with pytest.raises(ProductionBundleError, match="random_state"):
        _registered_contract_payload(
            {"name": "invalid", "odds_noise_seed": None, "xgb_params": {}}
        )


def test_integrity_contract_fallback_removes_only_six_method_features():
    full = full_live_contract_v6_integrity_211_spec()
    fallback = full_live_contract_v6_integrity_205_spec()

    assert len(full.feature_cols) == 211
    assert len(fallback.feature_cols) == 205
    assert set(full.feature_cols) - set(fallback.feature_cols) == set(
        V4_METHOD_ODDS_FEATURE_COLS
    )
    assert [
        column for column in full.feature_cols if column not in fallback.feature_cols
    ] == [
        column for column in full.feature_cols if column in V4_METHOD_ODDS_FEATURE_COLS
    ]
    for spec in (full, fallback):
        assert spec.odds_noise_mode == "antithetic"
        assert spec.odds_noise_std == pytest.approx(0.06)
        assert spec.odds_noise_seed == 42
        assert spec.xgb_params["random_state"] == 42


def test_integrity_fullfit_specs_only_advance_training_cutoff():
    assert full_live_contract_v6_integrity_211_fullfit_spec().feature_cols == (
        full_live_contract_v6_integrity_211_spec().feature_cols
    )
    assert full_live_contract_v6_integrity_205_fullfit_spec().feature_cols == (
        full_live_contract_v6_integrity_205_spec().feature_cols
    )
    assert full_live_contract_v6_integrity_211_fullfit_spec().train_cutoff_date == "2027-01-01"
    assert full_live_contract_v6_integrity_205_fullfit_spec().train_cutoff_date == "2027-01-01"


def test_live_roster_identity_uses_reviewed_modern_collision_id(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Jean Silva",
                "official_athlete_url": "https://www.ufc.com/athlete/jean-silva",
                "ufcstats_url": "http://ufcstats.com/fighter-details/9211aae062b799d6",
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(ufc_active_roster, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(live_monitor, "_ACTIVE_ROSTER_ID_CACHE", None)

    resolved = live_monitor._active_roster_ids_by_athlete_url()

    assert resolved["/athlete/jean-silva"] == "52ef95b5860fb28c"


def test_live_context_orients_fighter_ids_to_market_sides():
    context = bot._resolve_live_event_context(
        {
            "event_id": "event",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        },
        [
            {
                "event_id": "event",
                "fighter_a": "Beta Fighter",
                "fighter_b": "Alpha Fighter",
                "fighter_a_id": "bbbbbbbbbbbbbbbb",
                "fighter_b_id": "aaaaaaaaaaaaaaaa",
                "weight_class": "Lightweight",
                "is_title_bout": None,
                "num_rounds": None,
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert context is not None
    assert context["fighter_a_id"] == "aaaaaaaaaaaaaaaa"
    assert context["fighter_b_id"] == "bbbbbbbbbbbbbbbb"
    assert context["is_title_bout"] is None
    assert context["num_rounds"] is None


def test_prediction_context_cache_distinguishes_unknown_from_observed_false():
    fight = {"event_id": "event", "commence_time": "2026-08-15T22:00:00Z"}
    unknown = bot._prediction_event_context_snapshot(
        fight,
        {
            "weight_class": "Lightweight",
            "is_title_bout": None,
            "is_empty_arena": None,
            "num_rounds": None,
        },
    )
    observed_false = bot._prediction_event_context_snapshot(
        fight,
        {
            "weight_class": "Lightweight",
            "is_title_bout": False,
            "is_empty_arena": False,
            "num_rounds": 3,
        },
    )

    assert bot._LIVE_PREDICTION_INFERENCE_CONTRACT_VERSION == 2
    assert unknown["is_title_bout"] is None
    assert unknown != observed_false
