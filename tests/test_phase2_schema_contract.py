import json
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src import bot as bot_module
from src.data import fighter_lookup, historical_backfill, line_tracker, method_odds, rankings_scraper
from src.features import build_features as build_features_module
from src.model import predict as predict_module
from src.model import train as train_module
from src.model import training_spec
from src.strategy import backtest as backtest_module
from src.strategy import model_lab as model_lab_module
from src.strategy import model_variants as model_variants_module


def _nan_method_odds() -> dict:
    return {
        "a_ko_odds_prob": np.nan,
        "a_sub_odds_prob": np.nan,
        "a_dec_odds_prob": np.nan,
        "b_ko_odds_prob": np.nan,
        "b_sub_odds_prob": np.nan,
        "b_dec_odds_prob": np.nan,
    }


def _minimal_lookup_payload() -> dict:
    return {
        "features": {},
        "fights": [],
        "profile": {},
        "source": "test",
    }


def _minimal_training_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-01-01"),
                "winner": "Alpha",
                "target": 1,
                "a_elo": 1500.0,
                "b_elo": 1450.0,
                "a_num_fights": 2,
                "b_num_fights": 2,
            },
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-02-01"),
                "winner": "Beta",
                "target": 0,
                "a_elo": 1510.0,
                "b_elo": 1460.0,
                "a_num_fights": 3,
                "b_num_fights": 3,
            },
            {
                "fighter_a": "Gamma",
                "fighter_b": "Alpha",
                "event_date": pd.Timestamp("2022-02-01"),
                "winner": "Alpha",
                "target": 0,
                "a_elo": 1505.0,
                "b_elo": 1490.0,
                "a_num_fights": 2,
                "b_num_fights": 4,
            },
        ]
    )


class _CapturingModel:
    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = np.array(X, copy=True)
        return np.array([[0.4, 0.6] for _ in range(len(X))])


def test_line_movement_training_and_live_share_primary_logic(monkeypatch):
    historical = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 7,
                "a_fair_prob": 0.40,
                "b_fair_prob": 0.60,
            },
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 3,
                "a_fair_prob": 0.46,
                "b_fair_prob": 0.54,
            },
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 1,
                "a_fair_prob": 0.52,
                "b_fair_prob": 0.48,
            },
        ]
    )

    training_row = historical_backfill.compute_line_movement_from_backfill(historical).iloc[0]

    live_history = pd.DataFrame(
        {
            "snapshot_time": pd.to_datetime(
                ["2023-12-25T12:00:00", "2023-12-29T12:00:00", "2023-12-31T12:00:00"]
            ),
            "a_fair_prob": [0.40, 0.46, 0.52],
            "b_fair_prob": [0.60, 0.54, 0.48],
        }
    )
    monkeypatch.setattr(line_tracker, "load_line_history", lambda _a, _b, **_kwargs: live_history)

    live_features = line_tracker.get_line_movement_features("Alpha", "Beta")

    assert live_features["line_movement"] == pytest.approx(training_row["line_movement"])
    assert live_features["line_abs_movement"] == pytest.approx(training_row["line_abs_movement"])
    assert live_features["line_is_sharp"] == training_row["line_is_sharp"]
    assert live_features["line_steam_move"] == training_row["line_steam_move"]
    assert live_features["line_direction_toward_a"] == training_row["line_direction_toward_a"]
    assert live_features["line_direction_toward_b"] == training_row["line_direction_toward_b"]


def test_build_fight_features_missing_line_history_stays_nan(monkeypatch):
    monkeypatch.setattr(fighter_lookup, "lookup_fighter", lambda _name: _minimal_lookup_payload())
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(line_tracker, "analyze_line_movement", lambda *_args, **_kwargs: {"num_snapshots": 0})
    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {},
        "pfp": {},
        "source": "none",
        "acquisition_failed": True,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.60, "b_implied_prob": 0.40},
        weight_class="Lightweight",
    )

    assert np.isnan(features["line_movement"])
    assert np.isnan(features["line_abs_movement"])
    assert np.isnan(features["line_steam_move"])
    assert np.isnan(features["a_wc_rank_feat"])
    assert np.isnan(features["b_wc_rank_feat"])


def test_build_fight_features_strict_v3_uses_candidate_history_and_skips_excluded_live_calls(tmp_path, monkeypatch):
    spec = training_spec.full_live_contract_v3_spec()
    base_dir = tmp_path
    candidate_dir = tmp_path / "candidates" / spec.name
    candidate_dir.mkdir(parents=True)

    base_rows = [
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "winner": "Alpha",
            "weight_class": "Lightweight",
            "event_date": "2024-01-01",
            "a_roll_slpm": 99.0,
            "b_roll_slpm": 77.0,
            "a_roll_sapm": 2.0,
            "b_roll_sapm": 3.0,
            "a_roll_str_acc": 45.0,
            "b_roll_str_acc": 44.0,
            "a_roll_str_def": 55.0,
            "b_roll_str_def": 50.0,
            "a_roll_td_avg": 1.0,
            "b_roll_td_avg": 1.5,
            "a_roll_td_acc": 40.0,
            "b_roll_td_acc": 35.0,
            "a_roll_td_def": 70.0,
            "b_roll_td_def": 65.0,
            "a_roll_sub_avg": 0.2,
            "b_roll_sub_avg": 0.1,
            "a_roll_sig_str_landed": 40.0,
            "b_roll_sig_str_landed": 35.0,
            "a_roll_td_landed": 2.0,
            "b_roll_td_landed": 1.0,
            "a_roll_kd": 0.6,
            "b_roll_kd": 0.2,
            "a_roll_won": 0.8,
            "b_roll_won": 0.3,
            "a_elo": 1000.0,
            "b_elo": 900.0,
            "a_num_fights": 1,
            "b_num_fights": 1,
        },
        {
            "fighter_a": "Alpha",
            "fighter_b": "Gamma",
            "winner": "Alpha",
            "weight_class": "Lightweight",
            "event_date": "2024-02-01",
            "a_roll_slpm": 98.0,
            "b_roll_slpm": 76.0,
            "a_roll_sapm": 2.1,
            "b_roll_sapm": 3.1,
            "a_roll_str_acc": 46.0,
            "b_roll_str_acc": 43.0,
            "a_roll_str_def": 56.0,
            "b_roll_str_def": 49.0,
            "a_roll_td_avg": 1.2,
            "b_roll_td_avg": 1.4,
            "a_roll_td_acc": 41.0,
            "b_roll_td_acc": 34.0,
            "a_roll_td_def": 71.0,
            "b_roll_td_def": 64.0,
            "a_roll_sub_avg": 0.25,
            "b_roll_sub_avg": 0.12,
            "a_roll_sig_str_landed": 41.0,
            "b_roll_sig_str_landed": 34.0,
            "a_roll_td_landed": 2.1,
            "b_roll_td_landed": 1.1,
            "a_roll_kd": 0.7,
            "b_roll_kd": 0.3,
            "a_roll_won": 0.85,
            "b_roll_won": 0.25,
            "a_elo": 1100.0,
            "b_elo": 880.0,
            "a_num_fights": 2,
            "b_num_fights": 1,
        },
    ]
    candidate_rows = [
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "winner": "Alpha",
            "weight_class": "Lightweight",
            "event_date": "2024-01-01",
            "a_roll_slpm": 4.1,
            "b_roll_slpm": 3.4,
            "a_roll_sapm": 2.0,
            "b_roll_sapm": 3.0,
            "a_roll_str_acc": 45.0,
            "b_roll_str_acc": 44.0,
            "a_roll_str_def": 55.0,
            "b_roll_str_def": 50.0,
            "a_roll_td_avg": 1.0,
            "b_roll_td_avg": 1.5,
            "a_roll_td_acc": 40.0,
            "b_roll_td_acc": 35.0,
            "a_roll_td_def": 70.0,
            "b_roll_td_def": 65.0,
            "a_roll_sub_avg": 0.2,
            "b_roll_sub_avg": 0.1,
            "a_roll_sig_str_landed": 40.0,
            "b_roll_sig_str_landed": 35.0,
            "a_roll_td_landed": 2.0,
            "b_roll_td_landed": 1.0,
            "a_roll_kd": 0.6,
            "b_roll_kd": 0.2,
            "a_roll_won": 0.8,
            "b_roll_won": 0.3,
            "a_elo": 1500.0,
            "b_elo": 1400.0,
            "a_current_win_streak": 1.0,
            "b_current_win_streak": 0.0,
            "a_num_fights": 1,
            "b_num_fights": 1,
            "a_days_since_last_fight": 30.0,
            "b_days_since_last_fight": 45.0,
            "a_strike_diff": 2.1,
            "b_strike_diff": 0.4,
            "a_ko_rate": 0.5,
            "b_ko_rate": 0.2,
            "a_sub_rate": 0.2,
            "b_sub_rate": 0.1,
            "a_dec_rate": 0.3,
            "b_dec_rate": 0.7,
            "a_win_pct": 1.0,
            "b_win_pct": 0.5,
            "is_title_bout": 0.0,
            "num_rounds_feat": 3.0,
            "a_lose_streak": 0.0,
            "b_lose_streak": 1.0,
            "a_longest_win_streak": 1.0,
            "b_longest_win_streak": 1.0,
            "a_total_rounds": 3.0,
            "b_total_rounds": 3.0,
            "a_title_bouts": 0.0,
            "b_title_bouts": 0.0,
            "a_draws": 0.0,
            "b_draws": 0.0,
            "a_cage_rust": 0.0,
            "b_cage_rust": 0.0,
            "a_layoff_log": 3.4,
            "b_layoff_log": 3.8,
            "a_fight_pace": 6.1,
            "b_fight_pace": 6.4,
            "a_ctrl_efficiency": 0.3,
            "b_ctrl_efficiency": 0.2,
            "a_adj_win_pct": 1.0,
            "b_adj_win_pct": 0.48,
        },
        {
            "fighter_a": "Alpha",
            "fighter_b": "Gamma",
            "winner": "Alpha",
            "weight_class": "Lightweight",
            "event_date": "2024-02-01",
            "a_roll_slpm": 4.2,
            "b_roll_slpm": 2.9,
            "a_roll_sapm": 2.1,
            "b_roll_sapm": 3.2,
            "a_roll_str_acc": 46.0,
            "b_roll_str_acc": 42.0,
            "a_roll_str_def": 56.0,
            "b_roll_str_def": 48.0,
            "a_roll_td_avg": 1.2,
            "b_roll_td_avg": 1.1,
            "a_roll_td_acc": 41.0,
            "b_roll_td_acc": 31.0,
            "a_roll_td_def": 71.0,
            "b_roll_td_def": 62.0,
            "a_roll_sub_avg": 0.25,
            "b_roll_sub_avg": 0.08,
            "a_roll_sig_str_landed": 41.0,
            "b_roll_sig_str_landed": 30.0,
            "a_roll_td_landed": 2.1,
            "b_roll_td_landed": 0.8,
            "a_roll_kd": 0.7,
            "b_roll_kd": 0.1,
            "a_roll_won": 0.85,
            "b_roll_won": 0.2,
            "a_elo": 1510.0,
            "b_elo": 1450.0,
            "a_current_win_streak": 2.0,
            "b_current_win_streak": 0.0,
            "a_num_fights": 2,
            "b_num_fights": 1,
            "a_days_since_last_fight": 20.0,
            "b_days_since_last_fight": 35.0,
            "a_strike_diff": 2.1,
            "b_strike_diff": -0.3,
            "a_ko_rate": 0.5,
            "b_ko_rate": 0.1,
            "a_sub_rate": 0.25,
            "b_sub_rate": 0.2,
            "a_dec_rate": 0.25,
            "b_dec_rate": 0.7,
            "a_win_pct": 1.0,
            "b_win_pct": 0.5,
            "is_title_bout": 0.0,
            "num_rounds_feat": 3.0,
            "a_lose_streak": 0.0,
            "b_lose_streak": 1.0,
            "a_longest_win_streak": 2.0,
            "b_longest_win_streak": 1.0,
            "a_total_rounds": 6.0,
            "b_total_rounds": 3.0,
            "a_title_bouts": 0.0,
            "b_title_bouts": 0.0,
            "a_draws": 0.0,
            "b_draws": 0.0,
            "a_cage_rust": 0.0,
            "b_cage_rust": 0.0,
            "a_layoff_log": 3.0,
            "b_layoff_log": 3.6,
            "a_fight_pace": 6.3,
            "b_fight_pace": 6.1,
            "a_ctrl_efficiency": 0.28,
            "b_ctrl_efficiency": 0.18,
            "a_adj_win_pct": 1.0033333333,
            "b_adj_win_pct": 0.49,
        },
    ]

    pd.DataFrame(base_rows).to_csv(base_dir / "features.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(candidate_dir / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", base_dir)
    monkeypatch.setattr(
        fighter_lookup,
        "get_line_movement_live",
        lambda *_args, **_kwargs: pytest.fail("strict v3 should not request line movement"),
    )
    monkeypatch.setattr(
        rankings_scraper,
        "get_fighter_rankings",
        lambda *_args, **_kwargs: pytest.fail("strict v3 should not request rankings"),
    )
    monkeypatch.setattr(
        method_odds,
        "get_method_odds",
        lambda *_args, **_kwargs: pytest.fail("strict v3 should not request method odds"),
    )
    fighter_lookup.clear_cache()

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Gamma",
        odds_features={"a_implied_prob": 0.6, "b_implied_prob": 0.4, "diff_implied_prob": 0.2},
        weight_class="Lightweight",
        training_spec=spec,
    )

    assert set(features) == set(spec.feature_cols)
    assert features["a_roll_slpm"] == pytest.approx(4.2)
    assert features["a_elo_momentum"] == pytest.approx(10.0)
    assert features["a_sos"] == pytest.approx((1400.0 + 1450.0) / 2.0)
    assert "a_implied_prob" not in features
    assert "is_empty_arena" not in features

    fighter_lookup.clear_cache()


def test_rankings_source_failure_returns_nan_but_true_unranked_is_16(monkeypatch):
    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {},
        "pfp": {},
        "source": "none",
        "acquisition_failed": True,
    })

    failed = rankings_scraper.get_fighter_rankings("Alpha Fighter", weight_class="Lightweight")
    assert np.isnan(failed["wc_rank_feat"])
    assert np.isnan(failed["pfp_rank_feat"])

    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {"lightweight": {"beta gamma": 1}},
        "pfp": {"champ champ": 1},
        "source": "ufc.com",
    })

    unranked = rankings_scraper.get_fighter_rankings("Alpha Fighter", weight_class="Lightweight")
    assert unranked["wc_rank_feat"] == rankings_scraper.UNRANKED_DEFAULT
    assert unranked["pfp_rank_feat"] == rankings_scraper.UNRANKED_DEFAULT


def test_materialize_contract_transforms_replaces_stale_default_fills_with_nan():
    features_df = pd.DataFrame([
        {
            "title_bout": np.nan,
            "empty_arena": np.nan,
            "num_rounds": np.nan,
            "a_wc_rank": np.nan,
            "b_wc_rank": np.nan,
            "a_pfp_rank": np.nan,
            "b_pfp_rank": np.nan,
            "is_title_bout": 0.0,
            "is_empty_arena": 0.0,
            "num_rounds_feat": 3.0,
            "a_wc_rank_feat": 16.0,
            "b_wc_rank_feat": 16.0,
            "a_pfp_rank_feat": 16.0,
            "b_pfp_rank_feat": 16.0,
            "diff_wc_rank": 0.0,
            "diff_pfp_rank": 0.0,
        }
    ])

    transformed = training_spec.materialize_contract_transforms(features_df)
    row = transformed.iloc[0]

    assert np.isnan(row["is_title_bout"])
    assert np.isnan(row["is_empty_arena"])
    assert np.isnan(row["num_rounds_feat"])
    assert np.isnan(row["a_wc_rank_feat"])
    assert np.isnan(row["b_wc_rank_feat"])
    assert np.isnan(row["a_pfp_rank_feat"])
    assert np.isnan(row["b_pfp_rank_feat"])
    assert np.isnan(row["diff_wc_rank"])
    assert np.isnan(row["diff_pfp_rank"])


def test_materialize_contract_transforms_preserves_observed_context_and_true_unranked_values():
    features_df = pd.DataFrame([
        {
            "title_bout": 1.0,
            "empty_arena": 0.0,
            "num_rounds": 5.0,
            "a_wc_rank": 16.0,
            "b_wc_rank": 7.0,
            "a_pfp_rank": 16.0,
            "b_pfp_rank": 5.0,
            "is_title_bout": 0.0,
            "is_empty_arena": 1.0,
            "num_rounds_feat": 3.0,
            "a_wc_rank_feat": np.nan,
            "b_wc_rank_feat": np.nan,
            "a_pfp_rank_feat": np.nan,
            "b_pfp_rank_feat": np.nan,
            "diff_wc_rank": np.nan,
            "diff_pfp_rank": np.nan,
        }
    ])

    transformed = training_spec.materialize_contract_transforms(features_df)
    row = transformed.iloc[0]

    assert row["is_title_bout"] == 1.0
    assert row["is_empty_arena"] == 0.0
    assert row["num_rounds_feat"] == 5.0
    assert row["a_wc_rank_feat"] == 16.0
    assert row["b_wc_rank_feat"] == 7.0
    assert row["diff_wc_rank"] == 9.0
    assert row["a_pfp_rank_feat"] == 16.0
    assert row["b_pfp_rank_feat"] == 5.0
    assert row["diff_pfp_rank"] == 11.0


def test_train_all_models_materializes_spec_transforms_before_selecting_spec_columns(tmp_path, monkeypatch):
    features_df = _minimal_training_features()
    spec = training_spec.NamedModelTrainingSpec(
        name="transform_contract_test",
        feature_cols=[
            "is_rematch",
            "h2h_record_diff",
            "a_elo_momentum",
            "b_elo_momentum",
            "diff_elo_momentum",
            "a_sos",
            "b_sos",
            "diff_sos",
        ],
        train_cutoff_date="2022-01-01",
        add_rematch_features=True,
        add_elo_momentum=True,
        add_strength_of_schedule=True,
    )

    xgb_calls: list[tuple[pd.DataFrame, list[str], dict]] = []

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        xgb_calls.append((train_df.copy(), list(feature_cols), dict(kwargs)))
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(_train_df, feature_cols):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(training_spec.NamedModelTrainingSpec, "save", lambda self, path: None)

    result = train_module.train_all_models(features_df, spec=spec)

    assert result["feature_cols"] == spec.feature_cols
    assert xgb_calls[0][1] == spec.feature_cols
    assert list(xgb_calls[0][0]["is_rematch"]) == [0, 1]
    assert set(spec.feature_cols).issubset(xgb_calls[0][0].columns)


def test_full_live_contract_spec_fails_on_missing_required_columns_after_transforms():
    features_df = _minimal_training_features()

    with pytest.raises(ValueError, match="Spec features missing from DataFrame after transforms"):
        train_module.train_all_models(features_df, spec=training_spec.full_live_contract_spec())


def test_train_all_models_refuses_dead_train_time_contract_columns():
    features_df = _minimal_training_features().assign(a_roll_sapm=np.nan)
    spec = training_spec.NamedModelTrainingSpec(
        name="dead_contract_guard",
        feature_cols=["a_roll_sapm"],
        dataset_variant="legacy_only",
        train_cutoff_date="2022-01-01",
    )

    with pytest.raises(ValueError, match="dead train-time contract columns"):
        train_module.train_all_models(features_df, spec=spec)


def test_train_all_models_populates_git_hash_in_saved_training_spec(monkeypatch, tmp_path):
    features_df = _minimal_training_features()
    spec = training_spec.NamedModelTrainingSpec(
        name="git_hash_capture",
        feature_cols=["a_elo"],
        dataset_variant="legacy_only",
        train_cutoff_date="2022-01-01",
    )

    def fake_train_xgboost(_train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(_train_df, feature_cols):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(train_module, "_resolve_repo_git_hash", lambda: "deadbeef")

    result = train_module.train_all_models(features_df, spec=spec)
    saved_spec = json.loads((tmp_path / "git_hash_capture_spec.json").read_text())

    assert result["spec"].git_hash == "deadbeef"
    assert saved_spec["git_hash"] == "deadbeef"
    assert result["xgboost"]["training_spec"]["git_hash"] == "deadbeef"
    assert result["xgboost_no_odds"]["training_spec"]["git_hash"] == "deadbeef"


def test_train_all_models_embeds_filtered_contract_for_no_odds_artifact(monkeypatch, tmp_path):
    features_df = _minimal_training_features().assign(
        core_feature=[1.0, 0.5, -0.5],
        a_implied_prob=[0.55, 0.52, 0.48],
        b_implied_prob=[0.45, 0.48, 0.52],
        diff_implied_prob=[0.10, 0.04, -0.04],
    )
    spec = training_spec.NamedModelTrainingSpec(
        name="no_odds_contract_capture",
        feature_cols=["core_feature", "a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        dataset_variant="legacy_only",
        train_cutoff_date="2022-01-01",
    )

    def fake_train_xgboost(_train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(_train_df, feature_cols):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(training_spec.NamedModelTrainingSpec, "save", lambda self, path: None)

    result = train_module.train_all_models(features_df, spec=spec)

    assert result["xgboost"]["training_spec"]["feature_cols"] == spec.feature_cols
    assert result["xgboost_no_odds"]["training_spec"]["feature_cols"] == ["core_feature"]
    assert result["xgboost_no_odds"]["training_spec"]["name"] == "no_odds_contract_capture_no_odds"


def test_method_odds_zero_input_returns_nan():
    assert np.isnan(method_odds._american_to_implied_prob(0))
    assert np.isnan(method_odds._american_to_implied_prob(float("nan")))


def test_method_odds_matching_uses_full_names_not_last_name_substrings():
    event = {
        "home_team": "Bruno Silva",
        "away_team": "Anderson Silva",
        "bookmakers": [
            {
                "markets": [
                    {
                        "outcomes": [
                            {"name": "Bruno Silva by KO/TKO", "price": 200},
                            {"name": "Anderson Silva by Decision", "price": 150},
                        ]
                    }
                ]
            }
        ],
    }

    result = method_odds._extract_method_probs_from_event(
        event,
        home_is_a=True,
        fighter_a="Bruno Silva",
        fighter_b="Anderson Silva",
    )

    assert result is not None
    assert result["a_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(200))
    assert np.isnan(result["a_dec_odds_prob"])
    assert result["b_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(150))


def test_method_odds_name_match_rejects_partial_substrings():
    assert method_odds._names_match("Jon Jones", "Jon Jones")
    assert not method_odds._names_match("Jon", "Jonathan Martinez")
    assert not method_odds._names_match("Silva", "Bruno Silva")


def test_line_history_matching_requires_exact_normalized_fighter_names(tmp_path, monkeypatch):
    history_dir = tmp_path / "line_history"
    history_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fighter_a": "Bruno Silva",
                "fighter_b": "Anderson Silva",
                "snapshot_time": "2024-01-01T12:00:00",
                "a_fair_prob": 0.45,
                "b_fair_prob": 0.55,
            },
            {
                "fighter_a": "Bruno Silva Jr",
                "fighter_b": "Anderson Silva",
                "snapshot_time": "2024-01-01T13:00:00",
                "a_fair_prob": 0.50,
                "b_fair_prob": 0.50,
            },
        ]
    ).to_csv(history_dir / "odds_test.csv", index=False)

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", history_dir)

    history = line_tracker.load_line_history("Bruno Silva", "Anderson Silva")

    assert len(history) == 1
    assert history.iloc[0]["fighter_a"] == "Bruno Silva"


def test_historical_backfill_match_rejects_same_surname_false_positive():
    assert not historical_backfill._match_fighters(
        {"home_team": "Anderson Silva", "away_team": "Jon Jones"},
        "Bruno Silva",
        "Jon Jones",
    )


def test_merge_line_movement_features_handles_mixed_backfill_orientation():
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
            }
        ]
    )
    historical = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 7,
                "a_fair_prob": 0.40,
                "b_fair_prob": 0.60,
                "a_decimal_odds": 2.5,
                "b_decimal_odds": 1.6,
            },
            {
                "event_date": "2024-01-01",
                "fighter_a": "Beta",
                "fighter_b": "Alpha",
                "offset_days": 1,
                "a_fair_prob": 0.40,
                "b_fair_prob": 0.60,
                "a_decimal_odds": 2.5,
                "b_decimal_odds": 1.6,
            },
        ]
    )

    merged = historical_backfill.merge_line_movement_features(features_df, historical_df=historical)

    assert merged.loc[0, "line_movement"] == pytest.approx(0.20)
    assert merged.loc[0, "line_direction_toward_a"] == 1
    assert merged.loc[0, "line_direction_toward_b"] == 0


def test_predict_fight_native_nan_path_preserves_missing_values():
    model = _CapturingModel()

    result = predict_module.predict_fight(
        {"known_feature": 1.5},
        model_result={
            "model": model,
            "feature_cols": ["known_feature", "missing_feature"],
            "col_medians": np.array([0.0, 0.0]),
            "impute_strategy": "native_nan",
        },
    )

    assert result["prob_a"] == pytest.approx(0.6)
    assert model.last_X.shape == (1, 2)
    assert model.last_X[0, 0] == pytest.approx(1.5)
    assert np.isnan(model.last_X[0, 1])


def test_predict_batch_native_nan_path_fills_missing_columns_with_nan():
    model = _CapturingModel()

    result = predict_module.predict_batch(
        pd.DataFrame([{"known_feature": 1.5}]),
        model_result={
            "model": model,
            "feature_cols": ["known_feature", "missing_feature"],
            "col_medians": np.array([0.0, 0.0]),
            "impute_strategy": "native_nan",
        },
    )

    assert result["prob_a"].iloc[0] == pytest.approx(0.6)
    assert model.last_X.shape == (1, 2)
    assert model.last_X[0, 0] == pytest.approx(1.5)
    assert np.isnan(model.last_X[0, 1])


def test_model_lab_native_nan_path_matches_batch_inference_contract():
    model = _CapturingModel()

    result = model_lab_module._predict_batch_with_model(
        pd.DataFrame([{"known_feature": 1.5}]),
        {
            "model": model,
            "feature_cols": ["known_feature", "missing_feature"],
            "col_medians": np.array([0.0, 0.0]),
            "impute_strategy": "native_nan",
            "n_indicator_cols": 0,
        },
    )

    assert result["prob_a"].iloc[0] == pytest.approx(0.6)
    assert model.last_X.shape == (1, 2)
    assert model.last_X[0, 0] == pytest.approx(1.5)
    assert np.isnan(model.last_X[0, 1])


def test_cmd_train_uses_full_live_contract_spec(monkeypatch):
    captured = {}
    training_df = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-01-01"),
            }
        ]
    )

    monkeypatch.setattr(
        "src.data.kaggle_loader.load_kaggle_dataset",
        lambda *_args, **_kwargs: training_df.copy(),
    )
    monkeypatch.setattr(
        "src.data.ufc_refresh.build_training_dataset_variants",
        lambda *, legacy_df=None, **_kwargs: {"best_of_both_full_history": legacy_df.copy()},
    )
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.features.build_features.build_features", lambda fights_df: fights_df.assign(target=1))
    monkeypatch.setattr("src.features.build_features.save_features", lambda *_args, **_kwargs: None)

    def fake_train_all_models(features_df, spec=None):
        captured["spec"] = spec
        captured["features_df"] = features_df.copy()
        return {"train_df": features_df.iloc[:0], "test_df": features_df}

    monkeypatch.setattr("src.model.train.train_all_models", fake_train_all_models)

    bot_module.cmd_train(type("Args", (), {"data": None})())

    assert captured["spec"] is not None
    assert captured["spec"].name == "full_live_contract_v2"


def test_cmd_train_uses_spec_dataset_variant_when_no_explicit_data(monkeypatch):
    captured = {}
    legacy_df = pd.DataFrame([{"fighter_a": "Legacy", "fighter_b": "Only"}])
    variant_df = pd.DataFrame([{"fighter_a": "Variant", "fighter_b": "Rows"}])

    def fake_load_kaggle_dataset(filepath=None):
        captured.setdefault("load_paths", []).append(filepath)
        return legacy_df.copy()

    monkeypatch.setattr("src.data.kaggle_loader.load_kaggle_dataset", fake_load_kaggle_dataset)
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.data.ufc_refresh.build_training_dataset_variants",
        lambda *, legacy_df=None, **_kwargs: captured.setdefault("variant_inputs", []).append(legacy_df.copy()) or {
            "append_only_2026": variant_df.copy(),
        },
    )
    monkeypatch.setattr("src.features.build_features.build_features", lambda fights_df: fights_df.assign(target=1))
    monkeypatch.setattr("src.features.build_features.save_features", lambda *_args, **_kwargs: None)

    def fake_train_all_models(features_df, spec=None):
        captured["spec"] = spec
        captured["features_df"] = features_df.copy()
        return {"train_df": features_df.iloc[:0], "test_df": features_df}

    monkeypatch.setattr("src.model.train.train_all_models", fake_train_all_models)

    bot_module.cmd_train(
        type("Args", (), {"data": None, "training_spec": training_spec.append_only_2026_spec()})()
    )

    assert captured["spec"] is not None
    assert captured["spec"].dataset_variant == "append_only_2026"
    assert captured["load_paths"] == [bot_module.RAW_DATA_DIR / "ufc-master.csv"]
    assert captured["variant_inputs"][0].equals(legacy_df)
    assert captured["features_df"]["fighter_a"].tolist() == ["Variant"]


def test_cmd_train_resolves_named_spec_from_cli(monkeypatch):
    captured = {}
    training_df = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-01-01"),
            }
        ]
    )

    monkeypatch.setattr(
        "src.data.kaggle_loader.load_kaggle_dataset",
        lambda *_args, **_kwargs: training_df.copy(),
    )
    monkeypatch.setattr(
        "src.data.ufc_refresh.build_training_dataset_variants",
        lambda *, legacy_df=None, **_kwargs: {"pulled_all": legacy_df.copy()},
    )
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.features.build_features.build_features", lambda fights_df: fights_df.assign(target=1))
    monkeypatch.setattr("src.features.build_features.save_features", lambda *_args, **_kwargs: None)

    def fake_train_all_models(features_df, spec=None, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return {"train_df": features_df.iloc[:0], "test_df": features_df, "models_dir": bot_module.MODELS_DIR}

    monkeypatch.setattr("src.model.train.train_all_models", fake_train_all_models)

    bot_module.cmd_train(type("Args", (), {"data": None, "spec": "full_live_contract_v3"})())

    assert captured["spec"] is not None
    assert captured["spec"].name == "full_live_contract_v3"
    assert captured["spec"].dataset_variant == "pulled_all"


def test_cmd_train_output_subdir_writes_candidate_artifacts(monkeypatch):
    captured = {}
    training_df = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-01-01"),
            }
        ]
    )

    monkeypatch.setattr(
        "src.data.kaggle_loader.load_kaggle_dataset",
        lambda *_args, **_kwargs: training_df.copy(),
    )
    monkeypatch.setattr(
        "src.data.ufc_refresh.build_training_dataset_variants",
        lambda *, legacy_df=None, **_kwargs: {"pulled_all": legacy_df.copy()},
    )
    monkeypatch.setattr(
        "src.data.kaggle_loader.save_processed",
        lambda _df, filename="fights_cleaned.csv": captured.setdefault("processed", []).append(Path(filename)),
    )
    monkeypatch.setattr("src.features.build_features.build_features", lambda fights_df: fights_df.assign(target=1))
    monkeypatch.setattr(
        "src.features.build_features.save_features",
        lambda _df, filename="features.csv": captured.setdefault("features", []).append(Path(filename)),
    )

    def fake_train_all_models(features_df, spec=None, **kwargs):
        captured["spec"] = spec
        captured["train_kwargs"] = kwargs
        return {"train_df": features_df.iloc[:0], "test_df": features_df, "models_dir": kwargs.get("models_dir")}

    monkeypatch.setattr("src.model.train.train_all_models", fake_train_all_models)

    bot_module.cmd_train(
        type(
            "Args",
            (),
            {
                "data": None,
                "spec": "full_live_contract_v3",
                "output_subdir": "candidates/full_live_contract_v3",
            },
        )()
    )

    assert captured["processed"] == [Path("candidates/full_live_contract_v3/fights_cleaned.csv")]
    assert captured["features"] == [Path("candidates/full_live_contract_v3/features.csv")]
    assert captured["train_kwargs"]["models_dir"] == bot_module.MODELS_DIR / "candidates/full_live_contract_v3"
    assert captured["train_kwargs"]["test_set_path"] == (
        bot_module.PROCESSED_DATA_DIR / "candidates/full_live_contract_v3" / "test_set.csv"
    )


def test_cmd_train_explicit_data_path_overrides_dataset_variant(monkeypatch):
    captured = {}
    explicit_df = pd.DataFrame([{"fighter_a": "Explicit", "fighter_b": "Path"}])

    def fake_load_kaggle_dataset(filepath=None):
        captured.setdefault("load_paths", []).append(filepath)
        return explicit_df.copy()

    monkeypatch.setattr("src.data.kaggle_loader.load_kaggle_dataset", fake_load_kaggle_dataset)
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.data.ufc_refresh.build_training_dataset_variants",
        lambda **_kwargs: pytest.fail("dataset variants should not be built when --data is provided"),
    )
    monkeypatch.setattr("src.features.build_features.build_features", lambda fights_df: fights_df.assign(target=1))
    monkeypatch.setattr("src.features.build_features.save_features", lambda *_args, **_kwargs: None)

    def fake_train_all_models(features_df, spec=None):
        captured["spec"] = spec
        captured["features_df"] = features_df.copy()
        return {"train_df": features_df.iloc[:0], "test_df": features_df}

    monkeypatch.setattr("src.model.train.train_all_models", fake_train_all_models)

    bot_module.cmd_train(
        type(
            "Args",
            (),
            {
                "data": str(Path("C:/tmp/custom-train.csv")),
                "training_spec": training_spec.append_only_2026_spec(),
            },
        )()
    )

    assert captured["spec"] is not None
    assert captured["load_paths"] == [Path("C:/tmp/custom-train.csv")]
    assert captured["features_df"]["fighter_a"].tolist() == ["Explicit"]


def test_cmd_train_rejects_unknown_dataset_variant(monkeypatch):
    monkeypatch.setattr(
        "src.data.kaggle_loader.load_kaggle_dataset",
        lambda *_args, **_kwargs: pytest.fail("legacy dataset should not load for an unknown variant"),
    )
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)

    spec = training_spec.NamedModelTrainingSpec(name="bad_variant", dataset_variant="definitely_missing")

    with pytest.raises(ValueError, match="Unknown training dataset variant"):
        bot_module.cmd_train(type("Args", (), {"data": None, "training_spec": spec})())


def test_load_training_spec_from_artifact_prefers_embedded_spec(monkeypatch):
    embedded = training_spec.NamedModelTrainingSpec(
        name="embedded_spec",
        feature_cols=["a_elo"],
        add_rematch_features=True,
    )
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _model_name: {"training_spec": embedded.__dict__.copy()},
    )

    resolved = bot_module._load_training_spec_from_artifact("xgboost")

    assert resolved.name == "embedded_spec"
    assert resolved.feature_cols == ["a_elo"]


def test_load_training_spec_from_artifact_raises_on_invalid_embedded_spec(monkeypatch):
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _model_name: {"training_spec": {"name": "broken_spec", "unknown_flag": True}},
    )

    with pytest.raises(ValueError, match="Invalid embedded training spec"):
        bot_module._load_training_spec_from_artifact("xgboost")


def test_load_model_accepts_explicit_artifact_path():
    artifact_dir = Path(".pytest-artifacts") / uuid.uuid4().hex
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact_path = artifact_dir / "custom_model.pkl"
        payload = {
            "feature_cols": ["demo_feature"],
            "training_spec": {"feature_cols": ["demo_feature"]},
            "model": "stub-model",
        }
        joblib.dump(payload, artifact_path)

        loaded = train_module.load_model(str(artifact_path))

        assert loaded["feature_cols"] == ["demo_feature"]
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_load_model_repairs_legacy_no_odds_training_spec_contract():
    artifact_dir = Path(".pytest-artifacts") / uuid.uuid4().hex
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact_path = artifact_dir / "xgboost_no_odds_model.pkl"
        spec_feature_cols = ["core_feature", "a_implied_prob", "b_implied_prob", "diff_implied_prob"]
        payload = {
            "feature_cols": build_features_module.exclude_market_derived_features(spec_feature_cols),
            "training_spec": {
                "name": "legacy_promoted_spec",
                "description": "Legacy promoted spec",
                "feature_cols": spec_feature_cols,
            },
            "model": "stub-model",
        }
        joblib.dump(payload, artifact_path)

        loaded = train_module.load_model(str(artifact_path))

        assert loaded["feature_cols"] == ["core_feature"]
        assert loaded["training_spec"]["feature_cols"] == ["core_feature"]
        assert loaded["training_spec"]["name"] == "legacy_promoted_spec_no_odds"
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_load_model_rejects_artifact_missing_training_spec():
    artifact_dir = Path(".pytest-artifacts") / uuid.uuid4().hex
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact_path = artifact_dir / "custom_model.pkl"
        payload = {"feature_cols": ["demo_feature"], "model": "stub-model"}
        joblib.dump(payload, artifact_path)

        with pytest.raises(ValueError, match="missing an embedded training_spec"):
            train_module.load_model(str(artifact_path))
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_load_training_spec_from_artifact_rejects_missing_embedded_spec(monkeypatch):
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _model_name: {"feature_cols": ["a_elo"], "model": object()},
    )

    with pytest.raises(ValueError, match="does not embed a reproducible training spec"):
        bot_module._load_training_spec_from_artifact("xgboost")


def test_cmd_predict_passes_live_event_context_to_build_fight_features(monkeypatch):
    captured = []

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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    def fake_build_fight_features(*args, **kwargs):
        captured.append(kwargs)
        return {"a_num_fights": 4, "b_num_fights": 4}

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "col_medians": np.array([]), "feature_importance": {}, "raw_model": None},
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr("src.strategy.value._passes_filters", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_resolve_live_fight_counts", lambda *_args, **_kwargs: (4, 4))
    monkeypatch.setattr(
        bot_module,
        "_load_live_event_contexts",
        lambda: [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": True,
                "is_empty_arena": 1.0,
                "num_rounds": 5,
            }
        ],
    )

    bot_module.cmd_predict(type("Args", (), {"model": "xgboost"})())

    assert captured
    assert captured[0]["weight_class"] == "Lightweight"
    assert captured[0]["is_title_bout"] is True
    assert captured[0]["is_empty_arena"] == pytest.approx(1.0)
    assert captured[0]["num_rounds"] == 5


def test_cmd_predict_passes_embedded_training_spec_to_build_fight_features(monkeypatch):
    captured = []
    embedded = training_spec.full_live_contract_v3_spec()

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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    def fake_build_fight_features(*args, **kwargs):
        captured.append(kwargs)
        return {"a_num_fights": 4, "b_num_fights": 4}

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {
            "feature_cols": [],
            "col_medians": np.array([]),
            "feature_importance": {},
            "raw_model": None,
            "training_spec": embedded.__dict__.copy(),
        },
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_resolve_live_fight_counts", lambda *_args, **_kwargs: (4, 4))
    monkeypatch.setattr(
        bot_module,
        "_load_live_event_contexts",
        lambda: [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": True,
                "is_empty_arena": 1.0,
                "num_rounds": 5,
            }
        ],
    )

    bot_module.cmd_predict(type("Args", (), {"model": "xgboost"})())

    assert captured
    assert captured[0]["training_spec"].name == "full_live_contract_v3"


def test_resolve_live_event_context_fails_closed_on_blank_weight_class():
    event_context = bot_module._resolve_live_event_context(
        {
            "event_id": "evt-1",
            "commence_time": "2026-04-01T22:00:00Z",
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
        },
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "   ",
                "is_title_bout": True,
                "is_empty_arena": 1.0,
                "num_rounds": 5,
            },
            {
                "event_id": "evt-2",
                "commence_time": "2026-05-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": False,
                "is_empty_arena": 0.0,
                "num_rounds": 3,
            },
        ],
    )

    assert event_context is None


def test_resolve_live_event_context_preserves_empty_arena_flag():
    event_context = bot_module._resolve_live_event_context(
        {
            "event_id": "evt-1",
            "commence_time": "2026-04-01T22:00:00Z",
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
        },
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": False,
                "is_empty_arena": 1.0,
                "num_rounds": 3,
            }
        ],
    )

    assert event_context is not None
    assert event_context["is_empty_arena"] == pytest.approx(1.0)


def test_cmd_predict_skips_fights_without_live_event_context(monkeypatch):
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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "col_medians": np.array([]), "feature_importance": {}, "raw_model": None},
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr(
        "src.data.fighter_lookup.build_fight_features",
        lambda *_args, **_kwargs: pytest.fail("fight should be skipped before feature building"),
    )
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr("src.strategy.value._passes_filters", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_load_live_event_contexts", lambda: [])

    bot_module.cmd_predict(type("Args", (), {"model": "xgboost"})())


def test_cmd_predict_skips_fights_with_blank_weight_class_in_matched_context(monkeypatch):
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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "col_medians": np.array([]), "feature_importance": {}, "raw_model": None},
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr(
        "src.data.fighter_lookup.build_fight_features",
        lambda *_args, **_kwargs: pytest.fail("fight should be skipped before feature building"),
    )
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr("src.strategy.value._passes_filters", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bot_module,
        "_load_live_event_contexts",
        lambda: [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "   ",
                "is_title_bout": True,
                "num_rounds": 5,
            }
        ],
    )

    bot_module.cmd_predict(type("Args", (), {"model": "xgboost"})())


def test_cmd_duo_live_passes_live_event_context_to_build_fight_features(monkeypatch):
    captured = []

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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    def fake_build_fight_features(*args, **kwargs):
        captured.append(kwargs)
        return {"a_num_fights": 4, "b_num_fights": 4}

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "col_medians": np.array([]), "feature_importance": {}, "raw_model": None},
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
    monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr("src.polymarket.markets.get_ufc_fight_markets", lambda: pd.DataFrame([{"slug": "alpha-beta"}]))
    monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", lambda *_args, **_kwargs: {"total_orders": 0})
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_resolve_live_fight_counts", lambda *_args, **_kwargs: (4, 4))
    monkeypatch.setattr(
        bot_module,
        "_load_live_event_contexts",
        lambda: [
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-01T22:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": True,
                "is_empty_arena": 1.0,
                "num_rounds": 5,
            }
        ],
    )

    bot_module.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

    assert captured
    assert captured[0]["weight_class"] == "Lightweight"
    assert captured[0]["is_title_bout"] is True
    assert captured[0]["is_empty_arena"] == pytest.approx(1.0)
    assert captured[0]["num_rounds"] == 5


def test_cmd_duo_live_skips_fights_without_live_event_context(monkeypatch):
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
                        "commence_time": "2026-04-01T22:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "a_fair_prob_avg": 0.55,
                        "b_fair_prob_avg": 0.45,
                        "num_bookmakers": 8,
                    }
                ]
            )

    monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "col_medians": np.array([]), "feature_importance": {}, "raw_model": None},
    )
    monkeypatch.setattr("src.model.predict.predict_fight", lambda *_args, **_kwargs: {"prob_a": 0.6, "prob_b": 0.4, "confidence": 0.6})
    monkeypatch.setattr(
        "src.data.fighter_lookup.build_fight_features",
        lambda *_args, **_kwargs: pytest.fail("fight should be skipped before feature building"),
    )
    monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.data.line_tracker.detect_injury_or_cancellation", lambda *_args, **_kwargs: {"suspected": False})
    monkeypatch.setattr("src.polymarket.markets.get_ufc_fight_markets", lambda: pd.DataFrame([{"slug": "alpha-beta"}]))
    monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", lambda *_args, **_kwargs: {"total_orders": 0})
    monkeypatch.setattr(bot_module, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_load_live_event_contexts", lambda: [])

    bot_module.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())


def test_lookup_fighter_prefers_processed_feature_history(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "event_date": "2024-01-01",
                "winner": "Alpha Fighter",
                "a_elo": 1510.0,
                "a_wins": 4,
                "a_losses": 1,
                "a_draws": 0,
                "a_ko_rate": 0.25,
                "a_sub_rate": 0.10,
                "a_roll_str_def": 48.0,
                "a_roll_td_def": 55.0,
                "a_days_since_last_fight": 120.0,
            }
        ]
    ).to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        fighter_lookup,
        "search_fighter_url",
        lambda *_args, **_kwargs: pytest.fail("lookup_fighter should not hit UFCStats when processed history exists"),
    )
    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter("Alpha Fighter")

    assert result is not None
    assert result["source"] == "processed"
    assert result["features"]["elo"] == pytest.approx(1510.0)
    assert result["fights"][0]["opponent"] == "Beta Fighter"


def test_lookup_fighter_processed_reference_date_ages_snapshot_forward(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "event_date": "2024-01-01",
                "winner": "Alpha Fighter",
                "a_elo": 1510.0,
                "a_age": 30.0,
                "a_days_since_last_fight": 120.0,
                "a_layoff_log": float(np.log1p(120.0)),
                "a_cage_rust": 0.0,
            }
        ]
    ).to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        fighter_lookup,
        "search_fighter_url",
        lambda *_args, **_kwargs: pytest.fail("processed snapshot should be used before live scraping"),
    )
    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter(
        "Alpha Fighter",
        reference_date="2024-07-01T22:00:00Z",
    )

    assert result is not None
    assert result["source"] == "processed"
    assert result["features"]["days_since_last_fight"] == pytest.approx(182.0)
    assert result["features"]["layoff_log"] == pytest.approx(np.log1p(182.0))
    assert result["features"]["age"] == pytest.approx(30.0 + (182.0 / 365.25))

    fighter_lookup.clear_cache()


def test_call_lookup_fighter_preserves_supported_kwargs_while_retrying_unknown_ones(monkeypatch):
    def fake_lookup(name, as_of_date=None):
        return {
            "profile": {"name": name},
            "fights": [],
            "features": {"as_of_date": as_of_date},
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)

    result = fighter_lookup._call_lookup_fighter(
        "Alpha Fighter",
        as_of_date="2024-03-01",
        reference_date="2024-03-02T12:00:00Z",
        training_spec=training_spec.full_live_contract_v4_spec(),
        processed_data_dir=Path("dummy"),
    )

    assert result is not None
    assert result["features"]["as_of_date"] == "2024-03-01"


def test_lookup_fighter_historical_miss_fails_closed_without_live_scrape(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": "2024-06-01",
                "winner": "Alpha",
                "a_elo": 1510.0,
                "b_elo": 1490.0,
            }
        ]
    ).to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        fighter_lookup,
        "search_fighter_url",
        lambda *_args, **_kwargs: pytest.fail("historical lookup should fail closed before live scraping"),
    )
    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter("Alpha", as_of_date="2024-03-01")

    assert result is None

    fighter_lookup.clear_cache()


def test_lookup_fighter_strict_live_scrape_recovers_sparse_history_fields(monkeypatch):
    profile_html = """
    <html><body>
      <h2 class="b-content__title"><span>Beta Fighter</span></h2>
      <span class="b-content__title-record">Record: 1-0-0</span>
      <li class="b-list__box-list-item">Height: 5' 9"</li>
      <li class="b-list__box-list-item">Weight: 145 lbs.</li>
      <li class="b-list__box-list-item">Reach: 72"</li>
      <li class="b-list__box-list-item">STANCE: Orthodox</li>
      <li class="b-list__box-list-item">DOB: Apr 27, 1995</li>
      <tr class="b-fight-details__table-row" data-link="http://example.test/fight-details/1">
        <td><a class="b-flag">win</a></td>
        <td><p>Beta Fighter</p><p>Cam Teague</p></td>
        <td><p>1</p><p>0</p></td>
        <td><p>15 of 30</p><p>9 of 18</p></td>
        <td><p>1 of 2</p><p>0 of 1</p></td>
        <td><p>1</p><p>0</p></td>
        <td>DWCS 9.5 Sep. 09, 2025</td>
        <td>Decision - Unanimous</td>
        <td>3</td>
        <td>5:00</td>
      </tr>
    </body></html>
    """
    detail_html = """
    <html><body>
      <i class="b-fight-details__fight-title">Featherweight Bout</i>
      <table class="b-fight-details__table">
        <tbody>
          <tr class="b-fight-details__table-row">
            <td><p>Beta Fighter</p><p>Cam Teague</p></td>
            <td><p>1</p><p>0</p></td>
            <td><p>15 of 30</p><p>9 of 18</p></td>
            <td><p>50%</p><p>50%</p></td>
            <td><p>15 of 30</p><p>9 of 18</p></td>
            <td><p>1 of 2</p><p>0 of 1</p></td>
            <td><p>50%</p><p>0%</p></td>
            <td><p>1</p><p>0</p></td>
            <td><p>0</p><p>0</p></td>
            <td><p>2:30</p><p>0:45</p></td>
          </tr>
        </tbody>
      </table>
    </body></html>
    """

    soups = {
        "http://example.test/fighter": BeautifulSoup(profile_html, "lxml"),
        "http://example.test/fight-details/1": BeautifulSoup(detail_html, "lxml"),
    }

    monkeypatch.setattr(fighter_lookup, "search_fighter_url", lambda *_args, **_kwargs: "http://example.test/fighter")
    monkeypatch.setattr(fighter_lookup, "_get_soup", lambda url: soups[url])
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo", lambda *_args, **_kwargs: 1500.0)
    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter(
        "Beta Fighter",
        training_spec=training_spec.full_live_contract_v3_spec(),
    )

    assert result is not None
    assert result["source"] == "ufcstats"
    assert len(result["fights"]) == 1
    assert result["fights"][0]["event_date"] == pd.Timestamp("2025-09-09")
    assert result["fights"][0]["weight_class"] == "Featherweight Bout"
    assert result["features"]["roll_slpm"] == pytest.approx(1.0)
    assert result["features"]["roll_sapm"] == pytest.approx(0.6)
    assert result["features"]["roll_str_acc"] == pytest.approx(50.0)
    assert result["features"]["roll_str_def"] == pytest.approx(50.0)
    assert result["features"]["roll_td_avg"] == pytest.approx(1.0)
    assert result["features"]["roll_td_acc"] == pytest.approx(50.0)
    assert result["features"]["roll_td_def"] == pytest.approx(100.0)
    assert result["features"]["roll_sub_avg"] == pytest.approx(1.0)
    assert result["features"]["days_since_last_fight"] >= 0.0
    assert result["features"]["strike_diff"] == pytest.approx(0.4)
    assert fighter_lookup._compute_wc_move_from_history(result, "Featherweight") == pytest.approx(0.0)

    fighter_lookup.clear_cache()


def test_strict_live_sparse_history_preserves_honest_td_nan_rates():
    fights = [
        {
            "event_date": pd.Timestamp("2025-09-09"),
            "opponent": "Cam Teague",
            "won": 1,
            "result": "win",
            "winner": "Beta Fighter",
            "weight_class": "Featherweight Bout",
            "method": "KO/TKO Punch",
            "round_finished": 1,
            "slpm": 11.666666666666668,
            "sapm": 5.0,
            "str_acc": 100.0,
            "str_def": 0.0,
            "td_avg": 0.0,
            "td_acc": np.nan,
            "td_def": np.nan,
            "sub_avg": 0.0,
            "kd": 2.0,
            "sig_str_landed": 7.0,
            "sig_str_attempted": 7.0,
            "td_landed": 0.0,
            "td_attempted": 0.0,
            "sub_att": 0.0,
            "rev": 0.0,
            "ctrl_seconds": 4.0,
            "opp_kd": 0.0,
            "opp_sig_str_landed": 3.0,
            "opp_sig_str_attempted": 3.0,
            "opp_td_landed": 0.0,
            "opp_td_attempted": 0.0,
            "opp_sub_att": 0.0,
            "opp_rev": 0.0,
            "opp_ctrl_seconds": 0.0,
            "is_title_bout": False,
        }
    ]

    features = fighter_lookup._compute_rolling_for_fighter(
        fights,
        profile={"name": "Beta Fighter"},
        fighter_name="Beta Fighter",
        strict_mode=True,
    )

    assert features["roll_slpm"] == pytest.approx(11.666666666666668)
    assert features["roll_td_avg"] == pytest.approx(0.0)
    assert pd.isna(features["roll_td_acc"])
    assert pd.isna(features["roll_td_def"])


def test_live_wc_move_matches_training_semantics_for_historical_cutoff(tmp_path, monkeypatch):
    fights_df = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2024-01-01"),
                "winner": "Alpha",
                "weight_class": "Lightweight",
            },
            {
                "fighter_a": "Alpha",
                "fighter_b": "Gamma",
                "event_date": pd.Timestamp("2024-02-01"),
                "winner": "Alpha",
                "weight_class": "Lightweight",
            },
            {
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "event_date": pd.Timestamp("2024-03-01"),
                "winner": "Alpha",
                "weight_class": "Welterweight",
            },
        ]
    )
    training_features = build_features_module.build_features(fights_df)
    training_features.to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())
    fighter_lookup.clear_cache()

    live_features = fighter_lookup.build_fight_features(
        "Alpha",
        "Delta",
        odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5},
        weight_class="Welterweight",
        as_of_date="2024-03-01",
    )
    training_row = training_features.loc[
        (training_features["fighter_a"] == "Alpha")
        & (training_features["fighter_b"] == "Delta")
        & (training_features["event_date"] == pd.Timestamp("2024-03-01"))
    ].iloc[0]

    assert live_features["a_wc_move"] == pytest.approx(training_row["a_wc_move"])
    assert live_features["b_wc_move"] == pytest.approx(training_row["b_wc_move"])
    assert live_features["diff_wc_move"] == pytest.approx(training_row["diff_wc_move"])

    fighter_lookup.clear_cache()


def test_no_odds_feature_set_excludes_line_movement_columns():
    features_df = pd.DataFrame(
        [
            {
                "diff_elo": 25.0,
                "a_implied_prob": 0.55,
                "line_movement": 0.08,
                "line_abs_movement": 0.08,
                "line_is_sharp": 1,
            }
        ]
    )

    no_odds_cols = build_features_module.get_feature_columns_no_odds(features_df)

    assert "a_implied_prob" in build_features_module.ODDS_FEATURE_NAMES
    assert "line_movement" not in build_features_module.ODDS_FEATURE_NAMES
    assert "diff_elo" in no_odds_cols
    assert "a_implied_prob" not in no_odds_cols
    assert "line_movement" not in no_odds_cols
    assert "line_abs_movement" not in no_odds_cols
    assert "line_is_sharp" not in no_odds_cols


def test_build_fight_features_uses_commence_time_as_reference_date(monkeypatch):
    captured: list[dict] = []

    def fake_lookup(name, **kwargs):
        captured.append({"name": name, **kwargs})
        return _minimal_lookup_payload()

    monkeypatch.setattr(fighter_lookup, "_call_lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(line_tracker, "analyze_line_movement", lambda *_args, **_kwargs: {"num_snapshots": 0})
    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {},
        "pfp": {},
        "source": "none",
        "acquisition_failed": True,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.55, "b_implied_prob": 0.45},
        weight_class="Lightweight",
        commence_time="2026-04-01T22:00:00Z",
    )

    assert len(captured) == 2
    assert all(call["reference_date"] == "2026-04-01T22:00:00Z" for call in captured)


def test_live_processed_nan_defaults_match_training_semantics(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": "2024-03-01",
                "winner": "",
                "a_elo": 1500.0,
                "b_elo": 1500.0,
                "a_num_fights": 3,
                "b_num_fights": 3,
                "a_ko_rate": np.nan,
                "b_ko_rate": 0.2,
                "a_sub_rate": np.nan,
                "b_sub_rate": 0.05,
                "a_dec_rate": np.nan,
                "b_dec_rate": 0.3,
                "a_fight_pace": np.nan,
                "b_fight_pace": 4.0,
                "a_ctrl_efficiency": np.nan,
                "b_ctrl_efficiency": 0.2,
                "a_roll_str_def": np.nan,
                "b_roll_str_def": np.nan,
                "a_roll_td_def": np.nan,
                "b_roll_td_def": np.nan,
            }
        ]
    ).to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())
    fighter_lookup.clear_cache()

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5},
        as_of_date="2024-03-01",
    )

    assert features["a_ko_rate"] == pytest.approx(0.0)
    assert features["a_sub_rate"] == pytest.approx(0.0)
    assert features["a_dec_rate"] == pytest.approx(0.0)
    assert features["diff_ko_rate"] == pytest.approx(-0.2)
    assert features["diff_sub_rate"] == pytest.approx(-0.05)
    assert features["diff_dec_rate"] == pytest.approx(-0.3)
    assert features["a_fight_pace"] == pytest.approx(0.0)
    assert features["diff_fight_pace"] == pytest.approx(-4.0)
    assert features["a_ctrl_efficiency"] == pytest.approx(0.0)
    assert features["diff_ctrl_efficiency"] == pytest.approx(-0.2)
    assert features["a_striker_edge"] == pytest.approx(0.0)
    assert features["a_grappler_edge"] == pytest.approx(0.0)
    assert features["b_striker_edge"] == pytest.approx(0.1)
    assert features["b_grappler_edge"] == pytest.approx(0.025)

    fighter_lookup.clear_cache()


def test_live_rematch_detection_requires_exact_fighter_identity(monkeypatch):
    def fake_lookup(name):
        return {
            "profile": {"name": name, "record": "1-0-0"},
            "fights": [{"opponent": "Bruno Silva Jr", "won": 1, "event_date": pd.Timestamp("2024-01-01")}],
            "features": {
                "ko_rate": 0.2,
                "sub_rate": 0.1,
                "roll_str_def": np.nan,
                "roll_td_def": np.nan,
                "stance_enc": 0,
                "elo": 1500.0,
            },
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features("Alpha", "Bruno Silva", odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5})

    assert features["is_rematch"] == 0
    assert features["h2h_record_diff"] == 0


def test_live_same_stance_preserves_unknown_as_nan(monkeypatch):
    def fake_lookup(name):
        return {
            "profile": {"name": name, "record": "1-0-0"},
            "fights": [],
            "features": {
                "ko_rate": 0.2,
                "sub_rate": 0.1,
                "roll_str_def": np.nan,
                "roll_td_def": np.nan,
                "stance_enc": np.nan if name == "Alpha" else 0.0,
                "elo": 1500.0,
            },
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5},
    )

    assert pd.isna(features["same_stance"])


def test_live_rematch_detection_uses_b_history_when_a_history_misses(monkeypatch):
    def fake_lookup(name):
        fights = []
        if name == "Beta":
            fights = [
                {
                    "opponent": "Alpha",
                    "won": 1,
                    "result": "win",
                    "winner": "Beta",
                    "event_date": pd.Timestamp("2024-01-01"),
                }
            ]
        return {
            "profile": {"name": name, "record": "1-0-0"},
            "fights": fights,
            "features": {
                "ko_rate": 0.2,
                "sub_rate": 0.1,
                "roll_str_def": np.nan,
                "roll_td_def": np.nan,
                "stance_enc": 0,
                "elo": 1500.0,
            },
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5},
    )

    assert features["is_rematch"] == 1
    assert features["h2h_record_diff"] == -1


def test_live_h2h_treats_draw_or_no_contest_as_neutral(monkeypatch):
    def fake_lookup(name):
        fights = []
        if name == "Alpha":
            fights = [
                {
                    "opponent": "Beta",
                    "won": 0,
                    "result": "neutral",
                    "winner": "",
                    "event_date": pd.Timestamp("2024-01-01"),
                }
            ]
        return {
            "profile": {"name": name, "record": "1-0-0"},
            "fights": fights,
            "features": {
                "ko_rate": 0.2,
                "sub_rate": 0.1,
                "roll_str_def": np.nan,
                "roll_td_def": np.nan,
                "stance_enc": 0,
                "elo": 1500.0,
            },
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5},
    )

    assert features["is_rematch"] == 0
    assert features["h2h_record_diff"] == 0


def test_live_style_interaction_uses_training_fallback_for_missing_defense(monkeypatch):
    def fake_lookup(name):
        return {
            "profile": {"name": name, "record": "1-0-0"},
            "fights": [],
            "features": {
                "ko_rate": 0.2 if name == "Alpha" else 0.1,
                "sub_rate": 0.1 if name == "Alpha" else 0.05,
                "roll_str_def": np.nan,
                "roll_td_def": np.nan,
                "stance_enc": 0,
                "elo": 1500.0,
            },
            "source": "test",
        }

    monkeypatch.setattr(fighter_lookup, "lookup_fighter", fake_lookup)
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)
    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", lambda *args, **kwargs: {
        "line_movement": np.nan,
        "line_abs_movement": np.nan,
        "line_is_sharp": np.nan,
        "line_steam_move": np.nan,
        "line_direction_toward_a": np.nan,
        "line_direction_toward_b": np.nan,
    })
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", lambda *_args, **_kwargs: {
        "wc_rank_feat": np.nan,
        "pfp_rank_feat": np.nan,
    })
    monkeypatch.setattr(method_odds, "get_method_odds", lambda *_args, **_kwargs: _nan_method_odds())

    features = fighter_lookup.build_fight_features("Alpha", "Beta", odds_features={"a_implied_prob": 0.5, "b_implied_prob": 0.5})

    assert features["a_striker_edge"] == pytest.approx(0.1)
    assert features["a_grappler_edge"] == pytest.approx(0.05)


def test_elo_momentum_and_sos_respect_as_of_date(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "a_elo": 1500.0,
                "b_elo": 1400.0,
                "a_num_fights": 1,
                "b_num_fights": 1,
                "event_date": "2024-01-01",
            },
            {
                "fighter_a": "Gamma",
                "fighter_b": "Alpha",
                "a_elo": 1550.0,
                "b_elo": 1510.0,
                "a_num_fights": 3,
                "b_num_fights": 2,
                "event_date": "2024-02-01",
            },
            {
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "a_elo": 1600.0,
                "b_elo": 1450.0,
                "a_num_fights": 3,
                "b_num_fights": 1,
                "event_date": "2024-03-01",
            },
        ]
    ).to_csv(features_path, index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    fighter_lookup.clear_cache()

    momentum_before_march = fighter_lookup.get_fighter_elo_momentum("Alpha", as_of_date="2024-03-01")
    sos_before_march = fighter_lookup.get_fighter_sos("Alpha", as_of_date="2024-03-01")

    assert momentum_before_march == pytest.approx(10.0)
    assert sos_before_march == pytest.approx((1400.0 + 1550.0) / 2.0)

    fighter_lookup.clear_cache()


def test_compute_rolling_for_fighter_adj_win_pct_uses_as_of_date_filtered_sos(monkeypatch):
    captured = {}

    def fake_get_fighter_sos(fighter_name, window=5, as_of_date=None):
        captured["fighter_name"] = fighter_name
        captured["as_of_date"] = as_of_date
        return 1800.0

    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", fake_get_fighter_sos)

    rolling = fighter_lookup._compute_rolling_for_fighter(
        fights=[{"event_date": pd.Timestamp("2024-01-01"), "won": 1}],
        profile={
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "slpm": 4.0,
            "sapm": 2.0,
            "str_acc": 45.0,
            "str_def": 55.0,
            "td_avg": 1.0,
            "td_acc": 40.0,
            "td_def": 60.0,
            "sub_avg": 0.2,
        },
        fighter_name="Alpha",
        as_of_date="2024-03-01",
    )

    assert captured["fighter_name"] == "Alpha"
    assert captured["as_of_date"] == "2024-03-01"
    assert rolling["adj_win_pct"] == pytest.approx((3.0 / 4.0) * (1800.0 / 1500.0))


def test_build_fight_features_threads_as_of_date_into_external_snapshot_providers(monkeypatch):
    captured = {"line": [], "rank": [], "method": []}

    monkeypatch.setattr(
        fighter_lookup,
        "lookup_fighter",
        lambda *_args, **_kwargs: {
            "features": {"stance_enc": 0},
            "fights": [],
            "profile": {},
            "source": "test",
        },
    )
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo_momentum", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(fighter_lookup, "get_fighter_sos", lambda *args, **kwargs: 1500.0)

    def fake_line(*_args, **kwargs):
        captured["line"].append(kwargs.get("as_of_date"))
        return {
            "line_movement": np.nan,
            "line_abs_movement": np.nan,
            "line_is_sharp": np.nan,
            "line_steam_move": np.nan,
            "line_direction_toward_a": np.nan,
            "line_direction_toward_b": np.nan,
        }

    def fake_rankings(*_args, **kwargs):
        captured["rank"].append(kwargs.get("as_of_date"))
        return {"wc_rank_feat": np.nan, "pfp_rank_feat": np.nan}

    def fake_method(*_args, **kwargs):
        captured["method"].append(kwargs.get("as_of_date"))
        return _nan_method_odds()

    monkeypatch.setattr(fighter_lookup, "get_line_movement_live", fake_line)
    monkeypatch.setattr(rankings_scraper, "get_fighter_rankings", fake_rankings)
    monkeypatch.setattr(method_odds, "get_method_odds", fake_method)

    fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        odds_features={"a_implied_prob": 0.55, "b_implied_prob": 0.45},
        weight_class="Lightweight",
        as_of_date="2024-03-01",
    )

    assert captured["line"] == ["2024-03-01"]
    assert captured["rank"] == ["2024-03-01", "2024-03-01"]
    assert captured["method"] == ["2024-03-01"]


def test_promoted_specs_only_advertise_materialized_columns():
    features_path = fighter_lookup.PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        pytest.skip("processed features.csv not available")

    features_df = pd.read_csv(features_path, nrows=64, parse_dates=["event_date"])

    for spec_factory in [
        training_spec.rematch_features_spec,
        training_spec.full_live_contract_spec,
        training_spec.append_only_2026_spec,
    ]:
        spec = spec_factory()
        materialized = training_spec.materialize_spec_transforms(features_df, spec)
        missing = [column for column in spec.feature_cols if column not in materialized.columns]
        assert missing == [], spec.name


def test_full_live_contract_spec_uses_full_history_variant_without_line_movement():
    spec = training_spec.full_live_contract_spec()
    legacy_spec = training_spec.full_live_contract_v1_spec()

    assert spec.name == "full_live_contract_v2"
    assert spec.dataset_variant == "best_of_both_full_history"
    assert spec.add_line_movement is False
    assert "line_movement" not in spec.feature_cols
    assert legacy_spec.add_line_movement is True
    assert "line_movement" in legacy_spec.feature_cols


def test_full_live_contract_v3_spec_uses_pulled_all_and_excludes_legacy_only_families():
    spec = training_spec.full_live_contract_v3_spec()

    assert spec.name == "full_live_contract_v3"
    assert spec.dataset_variant == "pulled_all"
    assert spec.add_line_movement is False
    for excluded in [
        "a_height",
        "b_reach",
        "a_implied_prob",
        "diff_implied_prob",
        "is_empty_arena",
        "a_wc_rank_feat",
        "a_ko_odds_prob",
        "same_stance",
    ]:
        assert excluded not in spec.feature_cols
    for included in [
        "a_roll_sapm",
        "b_roll_str_def",
        "a_ko_rate",
        "diff_win_pct",
        "weight_class_enc",
        "is_title_bout",
        "is_rematch",
        "a_elo_momentum",
        "a_sos",
    ]:
        assert included in spec.feature_cols


def test_best_of_both_full_history_variant_backfills_pre_cutoff_overlap_fields():
    from src.data import ufc_refresh

    legacy = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2021-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "a_odds": -150.0,
                "title_bout": np.nan,
                "a_sapm": np.nan,
                "b_sapm": np.nan,
            }
        ]
    )
    pulled = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2021-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "a_odds": -110.0,
                "title_bout": 1.0,
                "a_sapm": 2.1,
                "b_sapm": 1.8,
            }
        ]
    )

    merged = ufc_refresh.build_best_of_both_full_history_training_dataset(legacy, pulled)

    assert len(merged) == 1
    assert merged.loc[0, "a_odds"] == pytest.approx(-150.0)
    assert merged.loc[0, "title_bout"] == pytest.approx(1.0)
    assert merged.loc[0, "a_sapm"] == pytest.approx(2.1)
    assert merged.loc[0, "b_sapm"] == pytest.approx(1.8)


def test_build_features_backfills_career_fields_from_history_when_raw_fields_missing():
    fights_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 1.0,
                "num_rounds": 3.0,
                "finish_round": 1.0,
                "method": "KO/TKO",
                "a_slpm": 2.0,
                "b_slpm": 1.0,
                "a_str_acc": 50.0,
                "b_str_acc": 40.0,
                "a_td_avg": 0.5,
                "b_td_avg": 0.2,
                "a_td_acc": 40.0,
                "b_td_acc": 25.0,
                "a_sub_avg": 0.1,
                "b_sub_avg": 0.0,
                "a_sapm": 1.0,
                "b_sapm": 2.0,
                "a_str_def": 55.0,
                "b_str_def": 45.0,
                "a_td_def": 70.0,
                "b_td_def": 50.0,
                "a_sig_str_landed": 30.0,
                "b_sig_str_landed": 20.0,
                "a_td_landed": 1.0,
                "b_td_landed": 0.0,
                "a_kd": 1.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 90.0,
                "b_ctrl_seconds": 30.0,
            },
            {
                "event_date": pd.Timestamp("2020-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Gamma",
                "winner": "Gamma",
                "target": 0,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "finish_round": 3.0,
                "method": "Decision - Unanimous",
                "a_slpm": 2.2,
                "b_slpm": 1.2,
                "a_str_acc": 48.0,
                "b_str_acc": 42.0,
                "a_td_avg": 0.4,
                "b_td_avg": 0.5,
                "a_td_acc": 38.0,
                "b_td_acc": 28.0,
                "a_sub_avg": 0.2,
                "b_sub_avg": 0.1,
                "a_sapm": 1.2,
                "b_sapm": 1.4,
                "a_str_def": 54.0,
                "b_str_def": 46.0,
                "a_td_def": 68.0,
                "b_td_def": 52.0,
                "a_sig_str_landed": 28.0,
                "b_sig_str_landed": 26.0,
                "a_td_landed": 0.0,
                "b_td_landed": 1.0,
                "a_kd": 0.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 80.0,
                "b_ctrl_seconds": 75.0,
            },
            {
                "event_date": pd.Timestamp("2021-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "finish_round": 2.0,
                "method": "Submission",
                "a_slpm": 2.4,
                "b_slpm": 1.0,
                "a_str_acc": 49.0,
                "b_str_acc": 39.0,
                "a_td_avg": 0.7,
                "b_td_avg": 0.1,
                "a_td_acc": 41.0,
                "b_td_acc": 20.0,
                "a_sub_avg": 0.4,
                "b_sub_avg": 0.0,
                "a_sapm": 0.8,
                "b_sapm": 1.8,
                "a_str_def": 58.0,
                "b_str_def": 44.0,
                "a_td_def": 72.0,
                "b_td_def": 48.0,
                "a_sig_str_landed": 32.0,
                "b_sig_str_landed": 18.0,
                "a_td_landed": 2.0,
                "b_td_landed": 0.0,
                "a_kd": 1.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 110.0,
                "b_ctrl_seconds": 20.0,
            },
        ]
    )

    features = build_features_module.build_features(fights_df)
    alpha_vs_delta = features.loc[features["fighter_b"] == "Delta"].iloc[0]

    assert alpha_vs_delta["a_wins"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_losses"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_draws"] == pytest.approx(0.0)
    assert alpha_vs_delta["a_lose_streak"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_longest_win_streak"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_total_rounds"] == pytest.approx(4.0)
    assert alpha_vs_delta["a_title_bouts"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_wins_ko"] == pytest.approx(1.0)
    assert alpha_vs_delta["a_wins_sub"] == pytest.approx(0.0)
    assert alpha_vs_delta["a_wins_dec"] == pytest.approx(0.0)
    assert alpha_vs_delta["a_win_pct"] == pytest.approx(0.5)



def test_model_lab_baseline_matches_promoted_contract():
    baseline = model_variants_module.baseline()
    production_spec = training_spec.full_live_contract_spec()
    production_no_odds = model_lab_module._production_no_odds_variant()

    assert baseline.feature_cols == production_spec.feature_cols
    assert baseline.calibration_method == production_spec.calibration_method
    assert baseline.calibration_cv == production_spec.calibration_cv
    assert baseline.add_rematch_features is True
    assert baseline.add_elo_momentum is True
    assert baseline.add_strength_of_schedule is True
    assert baseline.add_line_movement is False
    assert getattr(baseline, "_native_nan", False) is True
    assert production_no_odds.calibration_method == baseline.calibration_method
    assert production_no_odds.calibration_cv == baseline.calibration_cv
    assert getattr(production_no_odds, "_native_nan", False) is True


def test_model_lab_variants_inherit_promoted_contract_foundation(monkeypatch):
    captured = {}

    def fake_materialize(features_df, **kwargs):
        captured.update(kwargs)
        return features_df

    monkeypatch.setattr(model_lab_module, "materialize_contract_transforms", fake_materialize)

    model_lab_module._materialize_variant_contract_features(
        pd.DataFrame([{"fighter_a": "Alpha", "fighter_b": "Beta", "event_date": pd.Timestamp("2024-01-01")}]),
        model_variants_module.blend_b_fix(),
    )

    assert captured == {
        "add_rematch_features": True,
        "add_elo_momentum": True,
        "add_strength_of_schedule": True,
        "add_line_movement": False,
    }


def test_model_lab_delta_variants_inherit_promoted_feature_contract_and_native_nan():
    baseline = model_variants_module.baseline()

    blend_fix = model_lab_module._resolve_variant_against_promoted_baseline(
        model_variants_module.blend_b_fix()
    )
    temporal_sigmoid = model_lab_module._resolve_variant_against_promoted_baseline(
        model_variants_module.temporal_sigmoid_calibration()
    )
    missing_indicators = model_lab_module._resolve_variant_against_promoted_baseline(
        model_variants_module.missing_indicators()
    )

    assert blend_fix.feature_cols == baseline.feature_cols
    assert blend_fix.calibration_method == baseline.calibration_method
    assert blend_fix.calibration_cv == baseline.calibration_cv
    assert getattr(blend_fix, "_native_nan", False) is True

    assert temporal_sigmoid.feature_cols == baseline.feature_cols
    assert temporal_sigmoid.calibration_method == "sigmoid"
    assert temporal_sigmoid.calibration_cv == "temporal_holdout"
    assert getattr(temporal_sigmoid, "_native_nan", False) is True

    assert missing_indicators.feature_cols == baseline.feature_cols
    assert missing_indicators.impute_with_indicators is True
    assert getattr(missing_indicators, "_native_nan", False) is False


def test_model_lab_variant_feature_columns_fail_closed_on_missing_declared_columns(monkeypatch):
    features_df = pd.DataFrame([{"known_feature": 1.5}])

    with pytest.raises(ValueError, match="missing_feature"):
        model_lab_module._variant_feature_columns(
            features_df,
            model_variants_module.VariantConfig(
                name="custom_contract",
                description="test",
                feature_cols=["known_feature", "missing_feature"],
            ),
        )

    monkeypatch.setitem(
        model_lab_module.ALL_VARIANTS,
        "baseline",
        lambda: model_variants_module.VariantConfig(
            name="baseline",
            description="test",
            feature_cols=["known_feature", "missing_feature"],
        ),
    )

    with pytest.raises(ValueError, match="missing_feature"):
        model_lab_module._variant_feature_columns(
            features_df,
            model_variants_module.VariantConfig(
                name="inherits_baseline_contract",
                description="test",
            ),
        )


def test_merge_historical_odds_preserves_unsuffixed_line_filter_columns(monkeypatch):
    hist = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 7,
                "a_fair_prob": 0.55,
                "b_fair_prob": 0.45,
                "a_decimal_odds": 1.8,
                "b_decimal_odds": 2.1,
            },
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "offset_days": 1,
                "a_fair_prob": 0.60,
                "b_fair_prob": 0.40,
                "a_decimal_odds": 1.7,
                "b_decimal_odds": 2.2,
            },
        ]
    )
    movement = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "line_movement": 0.05,
                "line_abs_movement": 0.05,
                "line_is_sharp": 1,
                "line_steam_move": 0,
                "line_direction_toward_a": 1,
                "line_direction_toward_b": 0,
            }
        ]
    )

    monkeypatch.setattr("src.data.historical_backfill.load_historical_odds", lambda: hist)
    monkeypatch.setattr("src.data.historical_backfill.compute_line_movement_from_backfill", lambda _hist: movement)

    merged = backtest_module._merge_historical_odds(
        pd.DataFrame(
            [
                {
                    "event_date": pd.Timestamp("2024-01-01"),
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "line_movement": np.nan,
                    "line_is_sharp": np.nan,
                    "line_steam_move": np.nan,
                }
            ]
        )
    )

    assert "line_movement_x" not in merged.columns
    assert "line_movement_y" not in merged.columns
    assert merged.loc[0, "line_movement"] == pytest.approx(0.05)
    assert merged.loc[0, "line_is_sharp"] == pytest.approx(1)
    assert merged.loc[0, "line_steam_move"] == pytest.approx(0)


def test_historical_live_feature_vector_matches_full_contract_training_row(tmp_path, monkeypatch):
    reference_features_path = fighter_lookup.PROCESSED_DATA_DIR / "features.csv"
    reference_fights_path = fighter_lookup.PROCESSED_DATA_DIR / "fights_cleaned.csv"
    if not reference_features_path.exists() or not reference_fights_path.exists():
        pytest.skip("processed UFC data files not available")

    reference_features = pd.read_csv(reference_features_path, parse_dates=["event_date"])
    seed_row = reference_features[
        (reference_features["a_num_fights"] >= 3) & (reference_features["b_num_fights"] >= 3)
    ].iloc[0]

    fights_df = pd.read_csv(reference_fights_path, parse_dates=["event_date"])
    fights_subset = fights_df[fights_df["event_date"] <= seed_row["event_date"]].copy()
    rebuilt_features = build_features_module.build_features(fights_subset)
    spec = training_spec.full_live_contract_spec()
    rebuilt_features = training_spec.materialize_spec_transforms(rebuilt_features, spec)

    training_row = rebuilt_features.loc[
        (rebuilt_features["fighter_a"] == seed_row["fighter_a"])
        & (rebuilt_features["fighter_b"] == seed_row["fighter_b"])
        & (rebuilt_features["event_date"] == seed_row["event_date"])
    ].iloc[0]
    live_only_unknowns = [
        column
        for column in ["is_title_bout", "num_rounds_feat", "is_empty_arena"]
        if pd.isna(training_row.get(column, np.nan))
    ]
    if live_only_unknowns:
        pytest.skip(
            "selected historical row has unknown live-only event context: "
            + ", ".join(live_only_unknowns)
        )
    rebuilt_features.to_csv(tmp_path / "features.csv", index=False)

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        fighter_lookup,
        "get_line_movement_live",
        lambda *_args, **_kwargs: {
            "line_movement": training_row.get("line_movement", np.nan),
            "line_abs_movement": training_row.get("line_abs_movement", np.nan),
            "line_is_sharp": training_row.get("line_is_sharp", np.nan),
            "line_steam_move": training_row.get("line_steam_move", np.nan),
            "line_direction_toward_a": training_row.get("line_direction_toward_a", np.nan),
            "line_direction_toward_b": training_row.get("line_direction_toward_b", np.nan),
        },
    )
    monkeypatch.setattr(
        rankings_scraper,
        "get_fighter_rankings",
        lambda fighter_name, weight_class=None, **_kwargs: {
            "wc_rank_feat": training_row["a_wc_rank_feat"]
            if fighter_name == training_row["fighter_a"]
            else training_row["b_wc_rank_feat"],
            "pfp_rank_feat": training_row["a_pfp_rank_feat"]
            if fighter_name == training_row["fighter_a"]
            else training_row["b_pfp_rank_feat"],
        },
    )
    monkeypatch.setattr(
        method_odds,
        "get_method_odds",
        lambda *_args, **_kwargs: {
            "a_ko_odds_prob": training_row.get("a_ko_odds_prob", np.nan),
            "a_sub_odds_prob": training_row.get("a_sub_odds_prob", np.nan),
            "a_dec_odds_prob": training_row.get("a_dec_odds_prob", np.nan),
            "b_ko_odds_prob": training_row.get("b_ko_odds_prob", np.nan),
            "b_sub_odds_prob": training_row.get("b_sub_odds_prob", np.nan),
            "b_dec_odds_prob": training_row.get("b_dec_odds_prob", np.nan),
        },
    )
    fighter_lookup.clear_cache()

    live_features = fighter_lookup.build_fight_features(
        training_row["fighter_a"],
        training_row["fighter_b"],
        odds_features={
            "a_implied_prob": training_row["a_implied_prob"],
            "b_implied_prob": training_row["b_implied_prob"],
            "diff_implied_prob": training_row["diff_implied_prob"],
        },
        weight_class=training_row["weight_class"],
        is_title_bout=bool(training_row["is_title_bout"]),
        is_empty_arena=training_row["is_empty_arena"],
        num_rounds=int(training_row["num_rounds_feat"]),
        as_of_date=training_row["event_date"].strftime("%Y-%m-%d"),
    )

    mismatches = []
    for column in spec.feature_cols:
        expected = training_row.get(column, np.nan)
        actual = live_features.get(column, np.nan)
        if pd.isna(expected) and pd.isna(actual):
            continue
        if pd.isna(expected) != pd.isna(actual):
            mismatches.append(column)
            continue
        if not np.isclose(float(expected), float(actual), rtol=1e-6, atol=1e-6):
            mismatches.append(column)

    assert mismatches == []
    fighter_lookup.clear_cache()
