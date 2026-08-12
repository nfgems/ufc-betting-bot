import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import check_production_refit_contract as contract_gate
from scripts import check_scheduled_refit_quality as quality_gate
from scripts import recover_post_cutoff_method_odds as method_recovery
from scripts import track_c_batch
from src import bot as bot_module
from src.data import historical_backfill
from src.model import training_spec as training_spec_module
from src.strategy import backtest
from src.strategy import control_arm
from src.strategy import duo_trader_sweep
from src.strategy import run_evaluation
from src.strategy.promotion_gate import PromotionGate, evaluate_for_promotion


def _odds_row(offset, a_prob, *, source_kind="odds_api", verified=True):
    return {
        "event_date": "2026-06-06",
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "query_date": "2026-06-05",
        "offset_days": offset,
        "a_fair_prob": a_prob,
        "b_fair_prob": 1.0 - a_prob,
        "a_decimal_odds": 1.0 / a_prob,
        "b_decimal_odds": 1.0 / (1.0 - a_prob),
        "source_kind": source_kind,
        "source_file": f"{source_kind}.csv",
        "observed_at": pd.Timestamp("2026-06-06", tz="UTC") - pd.Timedelta(days=offset),
        "hours_to_start": offset * 24.0,
        "verified_prefight": verified,
    }


def test_role_selection_never_uses_offset_zero_as_opening_or_entry():
    history = pd.DataFrame(
        [
            _odds_row(7, 0.52),
            _odds_row(1, 0.57),
            _odds_row(0, 0.63, source_kind="bfo", verified=False),
        ]
    )

    roles = historical_backfill.build_historical_odds_roles(
        history, entry_offset_days=1.0
    ).iloc[0]

    assert roles["opening_prob_a"] == pytest.approx(0.52)
    assert roles["entry_prob_a"] == pytest.approx(0.57)
    assert roles["entry_source_kind"] == "odds_api"
    assert bool(roles["entry_verified_prefight"])
    assert roles["closing_prob_a"] == pytest.approx(0.63)
    assert roles["closing_source_kind"] == "bfo"
    assert not bool(roles["closing_verified_prefight"])


def test_line_history_importer_rejects_post_start_and_preserves_file(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "commence_time": "2026-06-07T02:00:00Z",
                "snapshot_time": "2026-06-06T00:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "a_odds": 1.8,
                "b_odds": 2.2,
                "a_fair_prob": 0.55,
                "b_fair_prob": 0.45,
            },
            {
                "commence_time": "2026-06-07T02:00:00Z",
                "snapshot_time": "2026-06-07T03:00:00Z",
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "a_odds": 1.7,
                "b_odds": 2.4,
                "a_fair_prob": 0.58,
                "b_fair_prob": 0.42,
            },
        ]
    )
    path = tmp_path / "odds_20260605_200000.csv"
    rows.to_csv(path, index=False)

    imported = historical_backfill.load_line_history_odds(tmp_path)

    assert len(imported) == 1
    assert imported.iloc[0]["source_kind"] == "line_history"
    assert imported.iloc[0]["source_file"] == path.name
    assert imported.iloc[0]["event_date"] == "2026-06-06"
    assert imported.iloc[0]["hours_to_start"] == pytest.approx(26.0)
    assert bool(imported.iloc[0]["verified_prefight"])


def test_scheduled_prediction_prep_clears_closing_and_scores_entry(monkeypatch):
    source = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-06-06"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "a_implied_prob": 0.99,
                "b_implied_prob": 0.01,
                "diff_implied_prob": 0.98,
                "a_fair_prob_avg": 0.98,
                "b_fair_prob_avg": 0.02,
            },
            {
                "event_date": pd.Timestamp("2026-06-13"),
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "a_implied_prob": 0.90,
                "b_implied_prob": 0.10,
                "diff_implied_prob": 0.80,
                "a_fair_prob_avg": 0.90,
                "b_fair_prob_avg": 0.10,
            },
        ]
    )

    def merge(frame, entry_offset_days=None):
        result = frame.copy()
        result["entry_prob_a"] = [0.56, np.nan]
        result["entry_prob_b"] = [0.44, np.nan]
        result["entry_source_kind"] = ["odds_api", ""]
        result["entry_source_file"] = ["historical_odds.csv", ""]
        result["entry_observed_at"] = [pd.Timestamp("2026-06-05", tz="UTC"), pd.NaT]
        result["entry_hours_to_start"] = [24.0, np.nan]
        result["entry_verified_prefight"] = [True, False]
        return result

    captured = {}

    def append_predictions(frame, model_name, model_result):
        captured[model_name] = frame.copy()
        frame[f"{model_name}_prob_a"] = 0.6
        frame[f"{model_name}_prob_b"] = 0.4
        frame[f"{model_name}_confidence"] = 0.6
        frame[f"{model_name}_predicted_winner"] = frame["fighter_a"]
        return frame

    monkeypatch.setattr(backtest, "_merge_historical_odds", merge)
    monkeypatch.setattr(backtest, "_append_model_predictions", append_predictions)

    prepared, _ = backtest._prepare_prediction_frame(
        source,
        model_results={"xgboost": {}},
        entry_offset_days=1.0,
        entry_offset_for_features=True,
        require_entry_odds=True,
    )

    scored = captured["xgboost"]
    assert scored.loc[0, "a_implied_prob"] == pytest.approx(0.56)
    assert np.isnan(scored.loc[1, "a_implied_prob"])
    assert bool(prepared.loc[0, "bet_eligible"])
    assert not bool(prepared.loc[1, "bet_eligible"])
    assert prepared.loc[0, "odds_source"] == "odds_api"


def test_policy_bound_fullfit_trainer_receives_evaluation_t_minus_one_semantics(
    monkeypatch,
    tmp_path,
):
    raw_features = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2026-06-06"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "a_implied_prob": 0.99,
                "b_implied_prob": 0.01,
                "diff_implied_prob": 0.98,
            },
            {
                "event_date": pd.Timestamp("2026-06-13"),
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "target": 0,
                "a_implied_prob": 0.90,
                "b_implied_prob": 0.10,
                "diff_implied_prob": 0.80,
            },
        ]
    )
    captured = {}

    monkeypatch.setattr(
        bot_module,
        "_load_training_dataframe",
        lambda **_kwargs: raw_features[["event_date", "fighter_a", "fighter_b", "target"]].copy(),
    )
    monkeypatch.setattr(
        "src.features.build_features.build_features",
        lambda _fights: raw_features.copy(),
    )
    monkeypatch.setattr("src.data.kaggle_loader.save_processed", lambda *_args, **_kwargs: None)

    def merge_roles(frame, entry_offset_days=None):
        captured.setdefault("entry_offsets", []).append(entry_offset_days)
        merged = frame.copy()
        merged["entry_prob_a"] = [0.56, np.nan]
        merged["entry_prob_b"] = [0.44, np.nan]
        merged["entry_source_kind"] = ["odds_api", ""]
        merged["entry_source_file"] = ["verified.csv", ""]
        merged["entry_observed_at"] = [pd.Timestamp("2026-06-05", tz="UTC"), pd.NaT]
        merged["entry_commence_time"] = [pd.Timestamp("2026-06-06", tz="UTC"), pd.NaT]
        merged["entry_hours_to_start"] = [24.0, np.nan]
        merged["entry_verified_prefight"] = [True, False]
        return merged

    monkeypatch.setattr(backtest, "_merge_historical_odds", merge_roles)
    monkeypatch.setattr(bot_module, "PROCESSED_DATA_DIR", tmp_path)

    def save_features(frame, filename="features.csv"):
        captured["saved_features"] = frame.copy()
        path = tmp_path / Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    monkeypatch.setattr("src.features.build_features.save_features", save_features)

    def train_all_models(frame, spec=None, **kwargs):
        captured["trainer_features"] = frame.copy()
        captured["training_spec"] = spec
        captured["training_kwargs"] = kwargs
        return {
            "train_df": frame,
            "test_df": frame.iloc[:0],
            "models_dir": kwargs.get("models_dir"),
        }

    monkeypatch.setattr("src.model.train.train_all_models", train_all_models)
    monkeypatch.setattr(
        "src.model.train._is_current_policy_selected_fullfit_spec",
        lambda _spec: True,
    )
    source_fights_path = tmp_path / "confirmed_fights_cleaned.csv"
    raw_features[["event_date", "fighter_a", "fighter_b", "target"]].to_csv(
        source_fights_path,
        index=False,
    )
    monkeypatch.setattr(
        "scripts.check_production_refit_contract.validate_final_track_c_pass_receipt",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "dataset_fights_path": source_fights_path.resolve(),
            "dataset_fights_sha256": bot_module._file_sha256(source_fights_path),
        },
    )
    spec = training_spec_module.full_live_contract_v6_integrity_205_fullfit_spec()

    bot_module.cmd_train(
        SimpleNamespace(
                data=str(source_fights_path),
            training_spec=spec,
            output_subdir="candidate",
            final_track_c_pass_receipt=tmp_path / "final_track_c_pass_receipt.json",
        )
    )

    saved = captured["saved_features"]
    trained = captured["trainer_features"]
    pd.testing.assert_frame_equal(trained, saved)
    assert captured["entry_offsets"] == [pytest.approx(1.0)]
    assert trained.loc[0, "a_implied_prob"] == pytest.approx(0.56)
    assert trained.loc[0, "b_implied_prob"] == pytest.approx(0.44)
    assert trained.loc[0, "diff_implied_prob"] == pytest.approx(0.12)
    assert np.isnan(trained.loc[1, "a_implied_prob"])
    assert np.isnan(trained.loc[1, "b_implied_prob"])
    assert np.isnan(trained.loc[1, "diff_implied_prob"])
    assert trained.loc[0, "model_odds_source_kind"] == "odds_api"
    assert bool(trained.loc[0, "model_odds_verified_prefight"])
    assert not bool(trained.loc[1, "model_odds_verified_prefight"])

    evidence = captured["training_kwargs"]["training_input_evidence"]
    assert evidence["policy_id"] == "honest-odds-weekly-refit-v2"
    assert evidence["entry_offset_days"] == pytest.approx(1.0)
    assert evidence["rows_with_verified_t_minus_entry"] == 1
    assert evidence["rows_missing_t_minus_entry"] == 1
    assert evidence["features_csv"]["path"] == str((tmp_path / "candidate/features.csv").resolve())
    assert len(evidence["features_csv"]["sha256"]) == 64


def test_unselected_integrity_fullfit_cannot_bypass_policy_binding():
    with pytest.raises(ValueError, match="is not selected by policy"):
        bot_module._load_policy_bound_fullfit_odds_config(
            training_spec_module.full_live_contract_v6_integrity_211_fullfit_spec()
        )


def test_explicit_closing_fallback_uses_closing_role_not_model_feature():
    frame = pd.DataFrame(
        [{
            "a_implied_prob": 0.55,
            "b_implied_prob": 0.45,
            "closing_prob_a": 0.70,
            "closing_prob_b": 0.30,
            "closing_source_kind": "bfo",
            "closing_source_file": "closing.csv",
            "closing_observed_at": "2026-06-06T00:00:00Z",
            "closing_hours_to_start": 0.0,
            "closing_verified_prefight": False,
        }]
    )

    resolved, _ = backtest._resolve_market_odds(
        frame,
        "a_fair_prob_avg",
        "b_fair_prob_avg",
        allow_closing_odds=True,
    )

    assert resolved.loc[0, "a_market_prob"] == pytest.approx(0.70)
    assert resolved.loc[0, "b_market_prob"] == pytest.approx(0.30)
    assert resolved.loc[0, "odds_source"] == "bfo"
    assert not bool(resolved.loc[0, "market_verified_prefight"])


def test_walkforward_preserves_row_level_odds_source(monkeypatch):
    from src.model import compare as compare_module
    from src.model import train as train_module
    from src.model import training_spec as spec_module

    spec = spec_module.NamedModelTrainingSpec(
        name="row_provenance_test",
        feature_cols=["signal"],
        impute_strategy="native_nan",
    )
    monkeypatch.setattr(
        spec_module,
        "materialize_and_validate_spec_features",
        lambda frame, _spec: frame,
    )
    monkeypatch.setattr(
        train_module,
        "train_xgboost",
        lambda frame, columns, **kwargs: {"feature_cols": list(columns)},
    )
    monkeypatch.setattr(backtest, "_prepare_evaluation_odds", lambda frame, **kwargs: frame.copy())

    def prepare_fold(frame, **kwargs):
        result = frame.copy()
        result["xgboost_prob_a"] = 0.60
        result["xgboost_prob_b"] = 0.40
        result["xgboost_no_odds_prob_a"] = 0.60
        result["xgboost_no_odds_prob_b"] = 0.40
        result["a_market_prob"] = 0.50
        result["b_market_prob"] = 0.50
        result["bet_eligible"] = True
        result["odds_source"] = np.where(result.index % 2, "line_history", "odds_api")
        return result, "aggregate-label"

    monkeypatch.setattr(backtest, "_prepare_prediction_frame", prepare_fold)
    monkeypatch.setattr(backtest, "_simulate_backtest_predictions", lambda *args, **kwargs: {})
    monkeypatch.setattr(backtest, "summarize_strategy_results", lambda results: pd.DataFrame())
    monkeypatch.setattr(backtest, "summarize_strategy_folds", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        backtest,
        "build_predictive_comparison_report",
        lambda frame: {"evaluation_window": {}, "models": {}, "comparison": {}},
    )
    monkeypatch.setattr(compare_module, "bootstrap_metric_comparison", lambda *args, **kwargs: {})

    dates = pd.date_range("2020-01-01", periods=220, freq="3D")
    features = pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A{i}" for i in range(len(dates))],
            "fighter_b": [f"B{i}" for i in range(len(dates))],
            "target": np.arange(len(dates)) % 2,
            "signal": 1.0,
        }
    )

    result = backtest.run_walkforward_strategy_comparison(
        features,
        retrain_months=6,
        initial_train_years=1,
        bet_start_date="2020-01-01",
        write_artifacts=False,
        spec=spec,
    )

    assert set(result["predictions"]["odds_source"]) == {"odds_api", "line_history"}
    assert "aggregate-label" not in set(result["predictions"]["odds_source"])


def test_data_quality_frame_is_one_to_one_and_records_method_coverage():
    prediction = pd.DataFrame(
        [
            {
                "event_date": "2026-06-06",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "fold": 1,
                "train_end": "2026-01-01",
                "test_end": "2026-07-01",
                "entry_source_kind": "odds_api",
                "entry_source_file": "historical_odds.csv",
                "entry_observed_at": "2026-06-05T00:00:00Z",
                "entry_hours_to_start": 24.0,
                "entry_verified_prefight": True,
                "model_odds_source_kind": "odds_api",
                "model_odds_source_file": "historical_odds.csv",
                "model_odds_observed_at": "2026-06-05T00:00:00Z",
                "model_odds_hours_to_start": 24.0,
                "model_odds_verified_prefight": True,
                "market_source_kind": "odds_api",
                "market_source_file": "historical_odds.csv",
                "market_observed_at": "2026-06-05T00:00:00Z",
                "market_hours_to_start": 24.0,
                "market_verified_prefight": True,
                "bet_eligible": True,
                **{column: 0.2 for column in track_c_batch._METHOD_ODDS_COLUMNS},
            }
        ]
    )

    evidence = track_c_batch._build_data_quality_frame(prediction)

    assert len(evidence) == 1
    assert bool(evidence.loc[0, "method_odds_complete"])
    assert evidence.loc[0, "entry_source_kind"] == "odds_api"
    assert evidence.loc[0, "fighter_a_identity_key"] == "name:alpha"
    assert evidence.loc[0, "fighter_b_identity_key"] == "name:beta"


def test_data_quality_csv_round_trip_materializes_missing_entry_verification_false(tmp_path):
    prediction = pd.DataFrame(
        [{
            "event_date": "2026-06-06", "fighter_a": "Alpha", "fighter_b": "Beta",
            "target": 1, "fold": 1, "train_end": "2026-01-01", "test_end": "2026-07-01",
            "entry_verified_prefight": np.nan,
        }]
    )

    evidence = track_c_batch._build_data_quality_frame(prediction)
    path = tmp_path / "quality.csv"
    evidence.to_csv(path, index=False, lineterminator="\n")
    loaded = pd.read_csv(path)

    assert evidence["entry_verified_prefight"].dtype == bool
    assert not bool(evidence.loc[0, "entry_verified_prefight"])
    assert not pd.isna(loaded.loc[0, "entry_verified_prefight"])
    assert quality_gate._strict_bool(
        loaded.loc[0, "entry_verified_prefight"],
        "data_quality.entry_verified_prefight",
    ) is False


def test_data_quality_materializes_nullable_boolean_missing_value_false():
    values = track_c_batch._explicit_boolean_series(
        pd.Series([True, pd.NA], dtype="boolean"),
        column="entry_verified_prefight",
    )

    assert values.dtype == bool
    assert values.tolist() == [True, False]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"entry_source_file": ""}, "missing entry_source_file"),
        ({"model_odds_observed_at": "not-a-time"}, "invalid model_odds_observed_at"),
        ({"market_hours_to_start": 999.0}, "does not match source timestamps"),
        ({"fighter_a": "Jean Silva"}, "missing fighter_a_identity_key"),
    ],
)
def test_quality_reader_rejects_incomplete_bound_provenance(tmp_path, mutation, message):
    prediction = pd.DataFrame(
        [{
            "event_date": "2026-06-06", "fighter_a": "Alpha", "fighter_b": "Beta",
            "target": 1, "fold": 1, "train_end": "2026-01-01", "test_end": "2026-07-01",
            "entry_source_kind": "odds_api", "entry_source_file": "odds.csv",
            "entry_observed_at": "2026-06-05T00:00:00Z", "entry_hours_to_start": 24.0,
            "entry_verified_prefight": True,
            "model_odds_source_kind": "odds_api", "model_odds_source_file": "odds.csv",
            "model_odds_observed_at": "2026-06-05T00:00:00Z",
            "model_odds_hours_to_start": 24.0, "model_odds_verified_prefight": True,
            "market_source_kind": "odds_api", "market_source_file": "odds.csv",
            "market_observed_at": "2026-06-05T00:00:00Z",
            "market_hours_to_start": 24.0, "market_verified_prefight": True,
            "bet_eligible": True,
            **mutation,
        }]
    )
    evidence = track_c_batch._build_data_quality_frame(prediction)
    path = tmp_path / "quality.csv"
    evidence.to_csv(path, index=False, lineterminator="\n")
    index = quality_gate.canonical_evaluation_index(
        evidence[quality_gate._EVALUATION_INDEX_COLUMNS]
    )

    with pytest.raises(quality_gate.QualityInputError, match=message):
        quality_gate._read_data_quality(
            path,
            expected_sha256=track_c_batch._sha256_file(path),
            evaluation_index=index,
        )


def test_quality_reader_recomputes_hours_from_source_timestamps(tmp_path):
    prediction = pd.DataFrame(
        [{
            "event_date": "2026-06-06", "fighter_a": "Alpha", "fighter_b": "Beta",
            "target": 1, "fold": 1, "train_end": "2026-01-01", "test_end": "2026-07-01",
            "entry_source_kind": "line_history", "entry_source_file": "odds_snapshot.csv",
            "entry_observed_at": "2026-06-05T02:00:00Z",
            "entry_commence_time": "2026-06-06T02:00:00Z",
            "entry_hours_to_start": np.nan, "entry_verified_prefight": True,
            "model_odds_source_kind": "line_history",
            "model_odds_source_file": "odds_snapshot.csv",
            "model_odds_observed_at": "2026-06-05T02:00:00Z",
            "model_odds_commence_time": "2026-06-06T02:00:00Z",
            "model_odds_hours_to_start": np.nan, "model_odds_verified_prefight": True,
            "market_source_kind": "line_history", "market_source_file": "odds_snapshot.csv",
            "market_observed_at": "2026-06-05T02:00:00Z",
            "market_commence_time": "2026-06-06T02:00:00Z",
            "market_hours_to_start": np.nan, "market_verified_prefight": True,
            "bet_eligible": True,
        }]
    )
    evidence = track_c_batch._build_data_quality_frame(prediction)
    path = tmp_path / "quality.csv"
    evidence.to_csv(path, index=False, lineterminator="\n")
    index = quality_gate.canonical_evaluation_index(
        evidence[quality_gate._EVALUATION_INDEX_COLUMNS]
    )

    loaded = quality_gate._read_data_quality(
        path,
        expected_sha256=track_c_batch._sha256_file(path),
        evaluation_index=index,
    )

    assert loaded.loc[0, "entry_hours_to_start"] == pytest.approx(24.0)
    assert loaded.loc[0, "model_odds_hours_to_start"] == pytest.approx(24.0)
    assert loaded.loc[0, "market_hours_to_start"] == pytest.approx(24.0)


def test_quality_reader_accepts_mixed_valid_utc_timestamp_formats(tmp_path):
    rows = []
    for suffix, observed_at in (
        ("One", "2026-06-05T00:00:00Z"),
        ("Two", "2026-06-05 00:00:00.123456+00:00"),
    ):
        row = {
            "event_date": "2026-06-06",
            "fighter_a": f"Alpha {suffix}",
            "fighter_b": f"Beta {suffix}",
            "target": 1,
            "fold": 1,
            "train_end": "2026-01-01",
            "test_end": "2026-07-01",
            "bet_eligible": True,
        }
        for prefix in ("entry", "model_odds", "market"):
            row.update(
                {
                    f"{prefix}_source_kind": "line_history",
                    f"{prefix}_source_file": "odds_snapshot.csv",
                    f"{prefix}_observed_at": observed_at,
                    f"{prefix}_commence_time": "2026-06-06T00:00:00Z",
                    f"{prefix}_hours_to_start": np.nan,
                    f"{prefix}_verified_prefight": True,
                }
            )
        rows.append(row)

    evidence = track_c_batch._build_data_quality_frame(pd.DataFrame(rows))
    path = tmp_path / "quality.csv"
    evidence.to_csv(path, index=False, lineterminator="\n")
    index = quality_gate.canonical_evaluation_index(
        evidence[quality_gate._EVALUATION_INDEX_COLUMNS]
    )

    loaded = quality_gate._read_data_quality(
        path,
        expected_sha256=track_c_batch._sha256_file(path),
        evaluation_index=index,
    )

    assert loaded[[
        "entry_verified_prefight",
        "model_odds_verified_prefight",
        "market_verified_prefight",
    ]].all().all()
    assert loaded.loc[1, "entry_hours_to_start"] == pytest.approx(
        23.9999657067
    )


def test_quality_artifact_hash_and_index_are_enforced(tmp_path):
    prediction = pd.DataFrame(
        [{
            "event_date": "2026-06-06", "fighter_a": "Alpha", "fighter_b": "Beta",
            "target": 1, "fold": 1, "train_end": "2026-01-01", "test_end": "2026-07-01",
        }]
    )
    evidence = track_c_batch._build_data_quality_frame(prediction)
    path = tmp_path / "quality.csv"
    evidence.to_csv(path, index=False, lineterminator="\n")
    digest = track_c_batch._sha256_file(path)
    index = quality_gate.canonical_evaluation_index(
        evidence[quality_gate._EVALUATION_INDEX_COLUMNS]
    )

    loaded = quality_gate._read_data_quality(
        path, expected_sha256=digest, evaluation_index=index
    )
    assert len(loaded) == 1

    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(quality_gate.QualityInputError, match="SHA-256 mismatch"):
        quality_gate._read_data_quality(
            path, expected_sha256=digest, evaluation_index=index
        )


def test_quality_gate_rejects_unverified_strategy_fill():
    dates = pd.date_range("2026-06-01", periods=50, freq="D")
    quality = pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A{i}" for i in range(50)],
            "fighter_b": [f"B{i}" for i in range(50)],
            "a_num_fights": [3] * 50,
            "b_num_fights": [3] * 50,
            "entry_source_kind": ["odds_api"] * 50,
            "entry_hours_to_start": [24.0] * 50,
            "entry_verified_prefight": [True] * 50,
            "model_odds_source_kind": ["odds_api"] * 50,
            "model_odds_hours_to_start": [24.0] * 50,
            "model_odds_verified_prefight": [True] * 50,
            "market_source_kind": ["odds_api"] * 49 + ["requested_feature"],
            "market_hours_to_start": [24.0] * 50,
            "market_verified_prefight": [True] * 49 + [False],
            "bet_eligible": [True] * 50,
            "method_odds_complete": [False] * 50,
        }
    )
    bets = quality.tail(1)[["event_date", "fighter_a", "fighter_b"]].copy()
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))

    errors, metrics = quality_gate._evaluate_provenance_quality(
        quality,
        bets,
        evaluation_end=dates.max(),
        policy=policy,
    )

    assert metrics["unverified_strategy_bets"] == 1
    assert any("lack verified T-1" in error for error in errors)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("market_source_kind", "bfo"),
        ("market_verified_prefight", False),
        ("market_hours_to_start", 23.0),
    ],
)
def test_quality_gate_rejects_unverified_eligible_non_bet(column, value):
    dates = pd.date_range("2026-06-01", periods=50, freq="D")
    quality = pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A{i}" for i in range(50)],
            "fighter_b": [f"B{i}" for i in range(50)],
            "fighter_a_identity_key": [f"name:a{i}" for i in range(50)],
            "fighter_b_identity_key": [f"name:b{i}" for i in range(50)],
            "a_num_fights": [3] * 50,
            "b_num_fights": [3] * 50,
            "bet_eligible": [True] * 50,
            "method_odds_complete": [False] * 50,
        }
    )
    for prefix in ("entry", "model_odds", "market"):
        quality[f"{prefix}_source_kind"] = "odds_api"
        quality[f"{prefix}_source_file"] = "historical_odds.csv"
        quality[f"{prefix}_observed_at"] = dates - pd.Timedelta(days=1)
        quality[f"{prefix}_hours_to_start"] = 24.0
        quality[f"{prefix}_verified_prefight"] = True
    quality.loc[quality.index[-1], column] = value
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))

    errors, metrics = quality_gate._evaluate_provenance_quality(
        quality,
        pd.DataFrame(),
        evaluation_end=dates.max(),
        policy=policy,
    )

    assert metrics["unverified_bet_eligible_rows"] == 1
    assert metrics["unverified_strategy_bets"] == 0
    assert any("bet-eligible rows lack verified T-1" in error for error in errors)


def test_quality_entry_lag_is_latest_event_freshness_not_snapshot_distance():
    quality = pd.DataFrame(
        {
            "event_date": [pd.Timestamp("2026-06-01")] * 50,
            "fighter_a": [f"A{i}" for i in range(50)],
            "fighter_b": [f"B{i}" for i in range(50)],
            "a_num_fights": [3] * 50,
            "b_num_fights": [3] * 50,
            "entry_source_kind": ["line_history"] * 50,
            # An early verified opening is still valid evidence.  The 35-day
            # gate measures how recently a fight with evidence occurred.
            "entry_hours_to_start": [40.0 * 24.0] * 50,
            "entry_verified_prefight": [True] * 50,
            "model_odds_source_kind": ["line_history"] * 50,
            "model_odds_hours_to_start": [40.0 * 24.0] * 50,
            "model_odds_verified_prefight": [True] * 50,
            "market_source_kind": ["line_history"] * 50,
            "market_hours_to_start": [40.0 * 24.0] * 50,
            "market_verified_prefight": [True] * 50,
            "bet_eligible": [True] * 50,
            "method_odds_complete": [False] * 50,
        }
    )
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))

    fresh_errors, fresh_metrics = quality_gate._evaluate_provenance_quality(
        quality,
        pd.DataFrame(),
        evaluation_end=pd.Timestamp("2026-06-01"),
        policy=policy,
    )
    assert not [error for error in fresh_errors if "behind evaluation end" in error]
    assert fresh_metrics["verified_entry_coverage"] == pytest.approx(1.0)
    assert fresh_metrics["latest_entry_event_date"] == "2026-06-01"
    assert fresh_metrics["latest_entry_lag_days"] == 0

    stale_errors, stale_metrics = quality_gate._evaluate_provenance_quality(
        quality,
        pd.DataFrame(),
        evaluation_end=pd.Timestamp("2026-07-10"),
        policy=policy,
    )
    assert stale_metrics["latest_entry_lag_days"] == 39
    assert any("39 days behind evaluation end" in error for error in stale_errors)


def test_policy_v2_locks_honest_odds_thresholds():
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))
    evaluation = policy["evaluation"]
    assert policy["contract"]["evaluation_spec_name"] == "full_live_contract_v6_integrity_205"
    assert policy["contract"]["feature_count"] == 205
    assert evaluation["require_entry_odds"] is True
    assert evaluation["quality_recent_window_days"] == 60
    assert evaluation["quality_minimum_recent_rows"] == 50
    assert evaluation["quality_minimum_entry_coverage"] == pytest.approx(0.70)
    assert evaluation["quality_maximum_entry_lag_days"] == pytest.approx(35.0)
    errors, _, _ = contract_gate.validate_policy_registry(policy)
    assert errors == []
    decision_path = Path(policy["contract"]["method_contract_decision_path"])
    assert contract_gate.file_sha256(decision_path) == policy["contract"][
        "method_contract_decision_sha256"
    ]
    assert quality_gate._validate_v2_baseline_evidence(policy) == []
    baseline_path = Path(policy["baseline"]["evidence_path"])
    assert contract_gate.file_sha256(baseline_path) == policy["baseline"][
        "evidence_sha256"
    ]


@pytest.mark.parametrize(
    ("sweep_payload", "message"),
    [
        ({"best_result": {"promotion_eligible": True}}, "requires a fixed production_result"),
        ({"production_result": {}}, "promotion_eligible=true"),
        ({"production_result": {"promotion_eligible": False}}, "promotion_eligible=true"),
    ],
)
def test_stage4_fails_closed_without_explicitly_eligible_production_result(
    monkeypatch, tmp_path, sweep_payload, message
):
    candidate_id = "candidate"
    monkeypatch.setattr(
        run_evaluation,
        "validate_frozen_control_arm_for_promotion_gate",
        lambda freeze_id: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(
        run_evaluation,
        "_load_control_arm_payloads",
        lambda freeze_id: (
            {"accuracy": 0.6, "brier": 0.24, "log_loss": 0.68, "ece": 0.05},
            {"roi": 0.1, "total_profit": 1.0, "max_drawdown": 0.1,
             "avg_clv": 0.01, "total_bets": 20},
        ),
    )
    finalists = [{
        "candidate_id": candidate_id,
        "model_variant": "combined_v3",
        "metrics": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04},
        "sliced_metrics": {"fresh_window": {}, "by_year": {}},
    }]

    with pytest.raises(ValueError, match=message):
        run_evaluation._run_stage4(
            run_dir=tmp_path,
            freeze_id="20260809",
            finalists=finalists,
            sweep_results={candidate_id: sweep_payload},
        )


def test_policy_v2_accepts_only_complete_method_decision(tmp_path, monkeypatch):
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))
    attempts = json.dumps(
        [
            {
                "pass": pass_name,
                "source": method_recovery._PASS_SOURCES[pass_name],
                "status": "no_evidence",
                "detail": "bounded pass completed without exact evidence",
            }
            for pass_name in method_recovery.RECOVERY_PASS_ORDER
        ]
    )
    terminal = pd.DataFrame(
        [
            {
                "target_key": f"target-{index}",
                "event_date": "2026-08-01",
                "established_fighters": True,
                "status": "failed",
                "terminal_reason": "no_exact_prefight_or_revalidated_evidence",
                "recovery_preconditions_met": True,
                "target_recovery_complete": True,
                "attempted_sources": attempts,
                "method_odds_complete": False,
                **{
                    column: np.nan
                    for column in method_recovery.METHOD_FEATURE_COLUMNS
                },
            }
            for index in range(12)
        ]
    )
    decision = method_recovery.decide_feature_contract(
        terminal, reference_date="2026-08-01"
    )
    path = tmp_path / "decision.json"
    monkeypatch.setattr(method_recovery, "REPO_ROOT", tmp_path)
    method_recovery.publish_outputs(
        tmp_path / "campaign",
        terminal,
        [],
        decision,
        canonical_decision_path=path,
    )
    policy["contract"]["method_contract_decision_path"] = path.name
    policy["contract"]["method_contract_decision_sha256"] = contract_gate.file_sha256(path)
    monkeypatch.setattr(contract_gate, "REPO_ROOT", tmp_path)

    errors, _, _ = contract_gate.validate_policy_registry(policy)

    assert not [error for error in errors if "method-contract" in error]

    backdated = method_recovery.decide_feature_contract(
        terminal, reference_date="2026-07-01"
    )
    method_recovery.publish_outputs(
        tmp_path / "campaign",
        terminal,
        [],
        backdated,
        canonical_decision_path=path,
    )
    policy["contract"]["method_contract_decision_sha256"] = (
        contract_gate.file_sha256(path)
    )

    errors, _, _ = contract_gate.validate_policy_registry(policy)

    assert any("does not reproduce" in error for error in errors)


def test_policy_v2_rejects_legacy_minimal_method_decision(tmp_path, monkeypatch):
    policy = contract_gate.load_policy(Path("config/scheduled_refit_policy_v2.json"))
    decision = {
        "schema_version": 1,
        "recovery_complete": True,
        "all_targets_terminal": True,
        "passes_completed": ["revalidate", "direct", "playwright", "wayback"],
        "targets_total": 12,
        "targets_terminal": 12,
        "selected_feature_count": 205,
        "selected_spec_name": "full_live_contract_v6_integrity_205",
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    policy["contract"]["method_contract_decision_path"] = path.name
    policy["contract"]["method_contract_decision_sha256"] = contract_gate.file_sha256(path)
    monkeypatch.setattr(contract_gate, "REPO_ROOT", tmp_path)

    errors, _, _ = contract_gate.validate_policy_registry(policy)

    assert any("method-contract decision" in error for error in errors)


def test_promotion_bootstrap_requires_twenty_events():
    dates = pd.date_range("2026-01-03", periods=19, freq="7D")
    candidate_log = pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A{i}" for i in range(19)],
            "fighter_b": [f"B{i}" for i in range(19)],
            "profit": [10.0] * 19,
            "bet_size": [10.0] * 19,
        }
    )
    baseline_log = candidate_log.assign(profit=0.0)
    result = PromotionGate.check_trading_bar(
        {"roi": 1.0, "total_profit": 190.0, "max_drawdown": 0.1, "bet_log": candidate_log},
        {"roi": 0.0, "total_profit": 0.0, "max_drawdown": 0.1, "bet_log": baseline_log},
    )
    assert not result["passes"]
    assert any("at least 20" in reason for reason in result["reasons"])


def test_standalone_promotion_requires_candidate_and_control_sample_hashes(monkeypatch):
    monkeypatch.setattr(
        PromotionGate,
        "check_model_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        PromotionGate,
        "check_trading_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )

    missing = evaluate_for_promotion({}, {}, {}, {})
    assert missing["verdict"] == "REJECT"
    assert any(
        "sample binding is incomplete" in reason
        for reason in missing["model_bar"]["reasons"]
    )

    bound = evaluate_for_promotion(
        {
            "evaluation_sample_sha256": "a" * 64,
            "evaluation_input_value_sha256": "b" * 64,
        },
        {
            "evaluation_sample_sha256": "a" * 64,
            "evaluation_input_value_sha256": "b" * 64,
        },
        {},
        {},
    )
    assert bound["verdict"] == "PROMOTE"


def test_general_stage3_generator_passes_honest_entry_policy(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        duo_trader_sweep,
        "_load_features_for_variant",
        lambda *args, **kwargs: pd.DataFrame({"event_date": []}),
    )
    monkeypatch.setattr(
        duo_trader_sweep,
        "ALL_VARIANTS",
        {"fixed": lambda: SimpleNamespace(calibration_method="isotonic")},
    )

    def generate(frame, variant, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(duo_trader_sweep, "generate_variant_fold_predictions", generate)

    duo_trader_sweep._generate_walk_forward_predictions(
        variant_name="fixed",
        entry_offset_days=1.0,
        entry_offset_for_features=True,
        require_entry_odds=True,
        allow_closing_odds=False,
    )

    assert captured["entry_offset_days"] == pytest.approx(1.0)
    assert captured["entry_offset_for_features"] is True
    assert captured["require_entry_odds"] is True
    assert captured["allow_closing_odds"] is False

    context = run_evaluation._stage3_context_from_payload(
        {
            "candidate_id": "fixed_candidate",
            "model_variant": "baseline",
            "dataset_variant": "append_only_2026",
            "feature_family": "production",
            "calibration_method": "isotonic",
            "retrain_months": 6,
        },
        run_narrow=False,
        metadata={
            "execution_mode": "realistic",
            "entry_offset_days": 1.0,
            "entry_offset_for_features": True,
            "require_entry_odds": True,
            "allow_closing_odds": False,
        },
    )
    assert context["entry_offset_days"] == pytest.approx(1.0)
    assert context["entry_offset_for_features"] is True
    assert context["require_entry_odds"] is True
    assert context["allow_closing_odds"] is False


def _write_control_run(root: Path, *, promotion_eligible: bool) -> Path:
    run_dir = root / "run"
    cells = run_dir / "cells"
    sweep = run_dir / "stage3_sweeps" / "candidate"
    cells.mkdir(parents=True)
    sweep.mkdir(parents=True)
    (cells / "candidate__isotonic_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.60,
                    "brier": 0.23,
                    "log_loss": 0.65,
                    "ece": 0.04,
                },
                "sliced_metrics": {},
                "evaluation_sample_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (sweep / "production_result.json").write_text(
        json.dumps(
            {
                "promotion_eligible": promotion_eligible,
                "roi": 0.05,
                "total_profit": 5.0,
                "total_bets": 1,
                "max_drawdown": 0.01,
                "avg_clv": 0.01,
                "source_marker": "production",
            }
        ),
        encoding="utf-8",
    )
    (sweep / "best_result.json").write_text(
        json.dumps({"promotion_eligible": False, "source_marker": "research"}),
        encoding="utf-8",
    )
    pd.DataFrame([{"profit": 5.0, "source_marker": "production"}]).to_csv(
        sweep / "production_bet_log.csv", index=False
    )
    pd.DataFrame([{"combined": 505.0}]).to_csv(
        sweep / "production_bankroll_history.csv", index=False
    )
    pd.DataFrame([{"profit": 999.0, "source_marker": "research"}]).to_csv(
        sweep / "best_bet_log.csv", index=False
    )
    return run_dir


def test_control_freezer_uses_only_fixed_production_artifacts(tmp_path, monkeypatch):
    run_dir = _write_control_run(tmp_path, promotion_eligible=True)
    frozen = tmp_path / "frozen"
    monkeypatch.setattr(control_arm, "FROZEN_DIR", frozen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_arm",
            "--run-dir", str(run_dir),
            "--candidate-id", "candidate",
            "--freeze-id", "20260809",
        ],
    )

    control_arm.main()

    destination = frozen / "control_arm_20260809"
    summary = json.loads((destination / "control_sweep_summary.json").read_text())
    metrics = json.loads((destination / "control_metrics.json").read_text())
    bets = pd.read_csv(destination / "control_bet_log.csv")
    assert summary["source_marker"] == "production"
    assert summary["promotion_eligible"] is True
    assert metrics["evaluation_sample_sha256"] == "a" * 64
    assert bets.loc[0, "source_marker"] == "production"


def test_control_freezer_rejects_research_only_result(tmp_path, monkeypatch):
    run_dir = _write_control_run(tmp_path, promotion_eligible=False)
    monkeypatch.setattr(control_arm, "FROZEN_DIR", tmp_path / "frozen")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_arm",
            "--run-dir", str(run_dir),
            "--candidate-id", "candidate",
            "--freeze-id", "20260809",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        control_arm.main()
    assert exc.value.code == 1
