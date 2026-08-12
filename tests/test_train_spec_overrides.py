from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.train_v4_tuned_candidate as tuned_candidate_script
from src.model import train as train_module
from src.model.training_spec import (
    NamedModelTrainingSpec,
    compute_feature_family_coverage,
    resolve_named_training_spec,
)


def _minimal_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "diff_stat": 0.8,
                "a_num_fights": 2,
                "b_num_fights": 2,
            },
            {
                "event_date": pd.Timestamp("2021-01-01"),
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "target": 0,
                "diff_stat": -0.7,
                "a_num_fights": 3,
                "b_num_fights": 3,
            },
            {
                "event_date": pd.Timestamp("2022-02-01"),
                "fighter_a": "Epsilon",
                "fighter_b": "Zeta",
                "target": 1,
                "diff_stat": 0.6,
                "a_num_fights": 4,
                "b_num_fights": 4,
            },
        ]
    )


def test_compute_sample_weights_accepts_half_life_override():
    train_df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21"]),
        }
    )

    baseline = train_module._compute_sample_weights(train_df, half_life_days=730)
    faster_decay = train_module._compute_sample_weights(train_df, half_life_days=30)

    assert baseline is not None
    assert faster_decay is not None
    assert faster_decay[0] < baseline[0]
    assert faster_decay[-1] > baseline[-1]


def test_train_all_models_threads_spec_decay_and_noise_overrides(tmp_path, monkeypatch):
    features_df = _minimal_features_df()
    spec = NamedModelTrainingSpec(
        name="override_smoke",
        feature_cols=["diff_stat"],
        train_cutoff_date="2022-01-01",
        add_rematch_features=False,
        add_line_movement=False,
        time_decay_half_life=123,
        odds_noise_std=0.07,
    )

    xgb_calls: list[dict] = []
    logistic_calls: list[dict] = []

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        xgb_calls.append(
            {
                "rows": len(train_df),
                "feature_cols": list(feature_cols),
                "kwargs": dict(kwargs),
            }
        )
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(train_df, feature_cols, **kwargs):
        logistic_calls.append(
            {
                "rows": len(train_df),
                "feature_cols": list(feature_cols),
                "kwargs": dict(kwargs),
            }
        )
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)

    result = train_module.train_all_models(
        features_df,
        spec=spec,
        models_dir=tmp_path / "models",
        test_set_path=tmp_path / "processed" / "test_set.csv",
    )

    assert result["spec"] is spec
    assert len(xgb_calls) == 2
    assert logistic_calls

    primary_xgb = xgb_calls[0]
    no_odds_xgb = xgb_calls[1]
    assert primary_xgb["feature_cols"] == ["diff_stat"]
    assert primary_xgb["kwargs"]["time_decay_half_life_days"] == 123
    assert primary_xgb["kwargs"]["odds_noise_std"] == pytest.approx(0.07)
    assert primary_xgb["kwargs"]["odds_noise_seed"] == 42
    assert no_odds_xgb["kwargs"]["time_decay_half_life_days"] == 123
    assert no_odds_xgb["kwargs"]["odds_noise_std"] == pytest.approx(0.07)
    assert no_odds_xgb["kwargs"]["odds_noise_seed"] == 42
    assert logistic_calls[0]["kwargs"]["odds_noise_std"] == pytest.approx(0.07)
    assert logistic_calls[0]["kwargs"]["odds_noise_seed"] == 42
    assert result["xgboost"]["training_spec"]["odds_noise_seed"] == 42
    assert (tmp_path / "processed" / "test_set.csv").exists()


def test_train_all_models_preserves_explicit_odds_noise_seed(tmp_path, monkeypatch):
    features_df = _minimal_features_df()
    spec = NamedModelTrainingSpec(
        name="seed_override_smoke",
        feature_cols=["diff_stat"],
        train_cutoff_date="2022-01-01",
        add_rematch_features=False,
        add_line_movement=False,
        odds_noise_seed=777,
    )

    xgb_calls: list[dict] = []
    logistic_calls: list[dict] = []

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        xgb_calls.append(dict(kwargs))
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(train_df, feature_cols, **kwargs):
        logistic_calls.append(dict(kwargs))
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)

    result = train_module.train_all_models(
        features_df,
        spec=spec,
        models_dir=tmp_path / "models",
        test_set_path=tmp_path / "processed" / "test_set.csv",
    )

    assert xgb_calls[0]["odds_noise_seed"] == 777
    assert xgb_calls[1]["odds_noise_seed"] == 777
    assert logistic_calls[0]["odds_noise_seed"] == 777
    assert result["xgboost"]["training_spec"]["odds_noise_seed"] == 777


def test_train_xgboost_rejects_unknown_calibration_cv():
    train_df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "target": [1, 0, 1],
            "diff_stat": [0.2, -0.1, 0.4],
        }
    )

    with pytest.raises(ValueError, match="Unsupported calibration_cv"):
        train_module.train_xgboost(
            train_df,
            ["diff_stat"],
            calibration_cv="definitely_not_supported",
            xgb_params={"random_state": 7},
        )


def test_train_xgboost_supports_temporal_holdout(monkeypatch):
    train_df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "target": [0, 1] * 5,
            "diff_stat": np.linspace(-1.0, 1.0, 10),
        }
    )

    xgb_instances: list[object] = []
    calibrators: list[object] = []

    class FakeXGB:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.feature_importances_ = np.array([1.0])
            self.fit_calls: list[dict] = []
            xgb_instances.append(self)

        def fit(self, X, y, sample_weight=None):
            self.fit_calls.append(
                {
                    "rows": len(X),
                    "sample_weight": None if sample_weight is None else np.asarray(sample_weight).copy(),
                }
            )
            return self

    class FakeCalibratedClassifierCV:
        def __init__(self, estimator, cv, method):
            self.estimator = estimator
            self.cv = cv
            self.method = method
            self.fit_calls: list[dict] = []
            calibrators.append(self)

        def fit(self, X, y, sample_weight=None):
            self.fit_calls.append({"rows": len(X), "sample_weight": sample_weight})
            return self

    monkeypatch.setattr(train_module, "XGBClassifier", FakeXGB)
    monkeypatch.setattr(train_module, "CalibratedClassifierCV", FakeCalibratedClassifierCV)
    monkeypatch.setattr(
        train_module,
        "_compute_sample_weights",
        lambda frame, half_life_days=None: np.ones(len(frame)),
    )

    result = train_module.train_xgboost(
        train_df,
        ["diff_stat"],
        calibration_method="sigmoid",
        calibration_cv="temporal_holdout",
        xgb_params={"random_state": 123},
        odds_noise_seed=123,
    )

    assert isinstance(result["model"], FakeCalibratedClassifierCV)
    assert len(xgb_instances) == 2
    assert xgb_instances[0].fit_calls[0]["rows"] == 10
    assert xgb_instances[1].fit_calls[0]["rows"] == 8
    assert isinstance(calibrators[0].estimator, train_module.FrozenEstimator)
    assert calibrators[0].fit_calls[0]["rows"] == 2


def test_compute_feature_family_coverage_requires_complete_rows():
    frame = pd.DataFrame(
        {
            "a_implied_prob": [0.6, np.nan, 0.4],
            "b_implied_prob": [0.4, 0.5, 0.6],
            "diff_implied_prob": [0.2, np.nan, -0.2],
        }
    )

    result = compute_feature_family_coverage(
        frame,
        feature_cols=["a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        family_names=["moneyline_odds"],
    )

    assert result["moneyline_odds"]["rows_total"] == 3
    assert result["moneyline_odds"]["rows_complete"] == 2
    assert result["moneyline_odds"]["coverage_pct"] == pytest.approx(66.67)


def test_train_all_models_rejects_insufficient_external_family_coverage(tmp_path):
    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "a_num_fights": 2,
                "b_num_fights": 2,
                "a_implied_prob": 0.60,
                "b_implied_prob": 0.40,
                "diff_implied_prob": 0.20,
            },
            {
                "event_date": pd.Timestamp("2021-01-01"),
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "target": 0,
                "a_num_fights": 2,
                "b_num_fights": 2,
                "a_implied_prob": np.nan,
                "b_implied_prob": 0.55,
                "diff_implied_prob": np.nan,
            },
            {
                "event_date": pd.Timestamp("2022-02-01"),
                "fighter_a": "Epsilon",
                "fighter_b": "Zeta",
                "target": 1,
                "a_num_fights": 3,
                "b_num_fights": 3,
                "a_implied_prob": 0.52,
                "b_implied_prob": 0.48,
                "diff_implied_prob": 0.04,
            },
        ]
    )
    spec = NamedModelTrainingSpec(
        name="coverage_gate_smoke",
        feature_cols=["a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        train_cutoff_date="2022-01-01",
        required_feature_family_coverage_pct={"moneyline_odds": 98.0},
        add_rematch_features=False,
        add_line_movement=False,
    )

    with pytest.raises(ValueError, match="failed required external-family coverage gates"):
        train_module.train_all_models(
            features_df,
            spec=spec,
            models_dir=tmp_path / "models",
            test_set_path=tmp_path / "processed" / "test_set.csv",
        )


def test_train_all_models_applies_training_window_bounds(tmp_path, monkeypatch):
    features_df = _minimal_features_df()
    spec = NamedModelTrainingSpec(
        name="window_smoke",
        feature_cols=["diff_stat"],
        train_cutoff_date="2022-01-01",
        train_start_date="2021-01-01",
        train_end_date="2021-06-01",
        add_rematch_features=False,
        add_line_movement=False,
    )

    observed_rows: list[int] = []

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        observed_rows.append(len(train_df))
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)

    train_module.train_all_models(
        features_df,
        spec=spec,
        models_dir=tmp_path / "models",
        test_set_path=tmp_path / "processed" / "test_set.csv",
    )

    assert observed_rows
    assert observed_rows[0] == 1


def test_train_all_models_writes_test_set_metadata(tmp_path, monkeypatch):
    features_df = _minimal_features_df()
    spec = NamedModelTrainingSpec(
        name="metadata_smoke",
        feature_cols=["diff_stat"],
        train_cutoff_date="2022-01-01",
        train_start_date="2014-01-01",
        dataset_variant="pulled_all_plus_legacy_market",
        add_rematch_features=False,
        add_line_movement=False,
    )

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)

    test_set_path = tmp_path / "processed" / "test_set.csv"
    result = train_module.train_all_models(
        features_df,
        spec=spec,
        models_dir=tmp_path / "models",
        test_set_path=test_set_path,
    )

    metadata_path = train_module.test_set_metadata_path(test_set_path)
    metadata = json.loads(metadata_path.read_text())

    assert metadata["spec_name"] == "metadata_smoke"
    assert metadata["feature_count"] == 1
    assert metadata["feature_hash"] == train_module._feature_contract_hash(["diff_stat"])
    assert metadata["dataset_variant"] == "pulled_all_plus_legacy_market"
    assert metadata["train_start_date"] == "2014-01-01"
    assert metadata["train_cutoff_date"] == "2022-01-01"
    assert metadata["row_count"] == len(result["test_df"])
    assert metadata["test_set_sha256"]


def test_train_all_models_embeds_policy_bound_training_input_evidence(
    tmp_path,
    monkeypatch,
):
    features_df = _minimal_features_df().assign(
        a_implied_prob=[0.56, np.nan, 0.52],
        b_implied_prob=[0.44, np.nan, 0.48],
        diff_implied_prob=[0.12, np.nan, 0.04],
        model_odds_source_kind=["odds_api", "", "line_history"],
        model_odds_source_file=["odds.csv", "", "line.csv"],
        model_odds_observed_at=["2020-01-01", "", "2022-01-31"],
        model_odds_commence_time=["2020-01-02", "", "2022-02-01"],
        model_odds_hours_to_start=[24.0, np.nan, 24.0],
        model_odds_verified_prefight=[True, False, True],
    )
    spec = NamedModelTrainingSpec(
        name="evidence_smoke",
        feature_cols=["diff_stat"],
        train_cutoff_date="2022-01-01",
        add_rematch_features=False,
        add_line_movement=False,
    )

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "raw_model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
            "impute_strategy": kwargs.get("impute_strategy", "native_nan"),
        }

    def fake_train_logistic(train_df, feature_cols, **kwargs):
        return {
            "model": None,
            "feature_cols": list(feature_cols),
            "feature_importance": {},
            "col_medians": np.array([]),
        }

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(train_module, "train_logistic", fake_train_logistic)
    monkeypatch.setattr(train_module.joblib, "dump", lambda *_args, **_kwargs: None)

    features_path = tmp_path / "features.csv"
    features_df.to_csv(features_path, index=False)
    source_fights_path = tmp_path / "source_fights.csv"
    features_df[["event_date", "target"]].to_csv(source_fights_path, index=False)
    payload = {
        "schema_version": 1,
        "preparation": "verified_t_minus_entry_model_odds",
        "policy_id": "policy-v2",
        "policy_path": str(tmp_path / "policy.json"),
        "policy_sha256": "a" * 64,
        "fullfit_spec_name": "selected-fullfit",
        "fullfit_spec_payload_sha256": "b" * 64,
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allowed_prefight_sources": ["line_history", "odds_api"],
        "row_count": 3,
        "rows_with_verified_t_minus_entry": 2,
        "rows_missing_t_minus_entry": 1,
        "prepared_odds_columns": [
            "a_implied_prob",
            "b_implied_prob",
            "diff_implied_prob",
        ],
        "provenance_columns": [
            "model_odds_source_kind",
            "model_odds_source_file",
            "model_odds_observed_at",
            "model_odds_commence_time",
            "model_odds_hours_to_start",
            "model_odds_verified_prefight",
        ],
        "features_csv": {
            "path": str(features_path.resolve()),
            "sha256": train_module._sha256_file(features_path),
            "bytes": features_path.stat().st_size,
        },
        "source_fights_csv": {
            "path": str(source_fights_path.resolve()),
            "sha256": train_module._sha256_file(source_fights_path),
            "bytes": source_fights_path.stat().st_size,
        },
    }
    evidence = {
        **payload,
        "receipt_sha256": train_module._canonical_json_sha256(payload),
    }
    test_set_path = tmp_path / "test_set.csv"

    tampered = features_df.copy()
    tampered.loc[0, "diff_stat"] = float(tampered.loc[0, "diff_stat"]) + 0.25
    with pytest.raises(
        ValueError,
        match="features CSV values/order differ from trainer input",
    ):
        train_module.train_all_models(
            tampered,
            spec=spec,
            models_dir=tmp_path / "tampered-models",
            test_set_path=tmp_path / "tampered-test-set.csv",
            training_input_evidence=evidence,
        )

    result = train_module.train_all_models(
        features_df,
        spec=spec,
        models_dir=tmp_path / "models",
        test_set_path=test_set_path,
        training_input_evidence=evidence,
    )

    for model_name in ("xgboost", "logistic", "xgboost_no_odds"):
        assert result[model_name]["training_input_evidence"] == evidence
    metadata = json.loads(train_module.test_set_metadata_path(test_set_path).read_text())
    assert metadata["training_input_evidence"] == evidence
    assert result["training_input_evidence"] == evidence


def test_integrity_fullfit_spec_requires_policy_bound_training_input_evidence(
):
    policy = json.loads(train_module._SCHEDULED_REFIT_POLICY_PATH.read_text())
    spec = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])

    with pytest.raises(
        ValueError,
        match="requires policy-bound training_input_evidence",
    ):
        train_module.train_all_models(pd.DataFrame(), spec=spec)


def test_policy_selected_fullfit_requires_final_policy_track_c_pass_receipt(
    monkeypatch,
):
    policy = json.loads(train_module._SCHEDULED_REFIT_POLICY_PATH.read_text())
    spec = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    monkeypatch.setattr(
        train_module, "_is_current_policy_selected_fullfit_spec", lambda _spec: True
    )

    with pytest.raises(
        ValueError,
        match="requires a final-policy Track-C PASS receipt",
    ):
        train_module.train_all_models(
            pd.DataFrame(),
            spec=spec,
            training_input_evidence={},
        )


@pytest.mark.parametrize("policy_bytes", [None, b"{not-json", b'{"schema_version":2}'])
def test_integrity_fullfit_policy_detection_fails_closed_without_blocking_research(
    tmp_path,
    monkeypatch,
    policy_bytes,
):
    broken_policy = tmp_path / "scheduled_refit_policy_v2.json"
    if policy_bytes is not None:
        broken_policy.write_bytes(policy_bytes)
    monkeypatch.setattr(train_module, "_SCHEDULED_REFIT_POLICY_PATH", broken_policy)
    integrity = resolve_named_training_spec(
        "full_live_contract_v6_integrity_205_fullfit"
    )
    research = NamedModelTrainingSpec(
        name="unrelated_research",
        feature_cols=["diff_stat"],
    )

    with pytest.raises(ValueError, match="cannot validate the scheduled policy"):
        train_module._is_current_policy_selected_fullfit_spec(integrity)
    assert train_module._is_current_policy_selected_fullfit_spec(research) is False


@pytest.mark.parametrize("decision_bytes", [None, b"{not-json"])
def test_integrity_family_detection_survives_missing_or_corrupt_method_decision(
    tmp_path,
    monkeypatch,
    decision_bytes,
):
    missing_policy = tmp_path / "missing-policy.json"
    broken_decision = tmp_path / "method-decision.json"
    if decision_bytes is not None:
        broken_decision.write_bytes(decision_bytes)
    monkeypatch.setattr(train_module, "_SCHEDULED_REFIT_POLICY_PATH", missing_policy)
    monkeypatch.setattr(train_module, "_METHOD_CONTRACT_DECISION_PATH", broken_decision)
    integrity = resolve_named_training_spec(
        "full_live_contract_v6_integrity_205_fullfit"
    )
    research = NamedModelTrainingSpec(
        name="unrelated_research",
        feature_cols=["diff_stat"],
    )

    with pytest.raises(ValueError, match="cannot validate the scheduled policy"):
        train_module._is_current_policy_selected_fullfit_spec(integrity)
    assert train_module._is_current_policy_selected_fullfit_spec(research) is False


def test_train_all_models_rejects_unselected_registered_integrity_fullfit():
    unselected = resolve_named_training_spec(
        "full_live_contract_v6_integrity_211_fullfit"
    )

    with pytest.raises(ValueError, match="is not selected by policy"):
        train_module.train_all_models(pd.DataFrame(), spec=unselected)


def test_train_all_models_rejects_renamed_selected_integrity_fullfit_clone():
    policy = json.loads(train_module._SCHEDULED_REFIT_POLICY_PATH.read_text())
    selected = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    renamed = replace(selected, name="renamed_integrity_fullfit_candidate")

    with pytest.raises(ValueError, match="is not selected by policy"):
        train_module.train_all_models(pd.DataFrame(), spec=renamed)


def test_integrity_evaluation_spec_remains_a_research_contract():
    evaluation = resolve_named_training_spec("full_live_contract_v6_integrity_205")

    assert train_module._matches_method_selected_fullfit_semantics(evaluation) is False
    assert train_module._is_current_policy_selected_fullfit_spec(evaluation) is False


def test_tuned_candidate_cli_rejects_unselected_integrity_fullfit(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v4_tuned_candidate.py",
            "--base-spec",
            "full_live_contract_v6_integrity_211_fullfit",
            "--output-subdir",
            "must-not-train",
        ],
    )
    monkeypatch.setattr(
        tuned_candidate_script,
        "_load_training_dataframe",
        lambda _spec: pd.DataFrame(),
    )
    monkeypatch.setattr(
        tuned_candidate_script,
        "build_features",
        lambda _frame: pd.DataFrame(),
    )
    monkeypatch.setattr(
        tuned_candidate_script,
        "save_processed",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tuned_candidate_script,
        "save_features",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="is not selected by policy"):
        tuned_candidate_script.main()


def _direct_training_input_evidence_fixture(tmp_path):
    features_df = _minimal_features_df().assign(
        a_implied_prob=[0.56, np.nan, 0.52],
        b_implied_prob=[0.44, np.nan, 0.48],
        diff_implied_prob=[0.12, np.nan, 0.04],
        model_odds_source_kind=["odds_api", "", "line_history"],
        model_odds_source_file=["odds.csv", "", "line.csv"],
        model_odds_observed_at=["2020-01-01", "", "2022-01-31"],
        model_odds_commence_time=["2020-01-02", "", "2022-02-01"],
        model_odds_hours_to_start=[24.0, np.nan, 24.0],
        model_odds_verified_prefight=[True, False, True],
    )
    policy_path = train_module._SCHEDULED_REFIT_POLICY_PATH
    policy = json.loads(policy_path.read_text())
    spec = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    spec_payload = asdict(spec)
    spec_payload["trained_at"] = ""
    spec_payload["git_hash"] = ""

    features_path = tmp_path / "features.csv"
    features_df.to_csv(features_path, index=False)
    source_fights_path = tmp_path / "source_fights.csv"
    features_df[["event_date", "target"]].to_csv(source_fights_path, index=False)
    payload = {
        "schema_version": 1,
        "preparation": "verified_t_minus_entry_model_odds",
        "policy_id": policy["policy_id"],
        "policy_path": str(policy_path),
        "policy_sha256": train_module._sha256_file(policy_path),
        "fullfit_spec_name": spec.name,
        "fullfit_spec_payload_sha256": train_module._canonical_json_sha256(
            spec_payload
        ),
        "entry_offset_days": policy["evaluation"]["entry_offset_days"],
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allowed_prefight_sources": policy["evaluation"][
            "quality_allowed_prefight_sources"
        ],
        "row_count": len(features_df),
        "rows_with_verified_t_minus_entry": 2,
        "rows_missing_t_minus_entry": 1,
        "prepared_odds_columns": [
            "a_implied_prob",
            "b_implied_prob",
            "diff_implied_prob",
        ],
        "provenance_columns": [
            "model_odds_source_kind",
            "model_odds_source_file",
            "model_odds_observed_at",
            "model_odds_commence_time",
            "model_odds_hours_to_start",
            "model_odds_verified_prefight",
        ],
        "features_csv": {
            "path": str(features_path.resolve()),
            "sha256": train_module._sha256_file(features_path),
            "bytes": features_path.stat().st_size,
        },
        "source_fights_csv": {
            "path": str(source_fights_path.resolve()),
            "sha256": train_module._sha256_file(source_fights_path),
            "bytes": source_fights_path.stat().st_size,
        },
    }
    evidence = {
        **payload,
        "receipt_sha256": train_module._canonical_json_sha256(payload),
    }
    return features_df, evidence, spec


def _resign_training_input_evidence(evidence: dict) -> dict:
    resigned = deepcopy(evidence)
    resigned.pop("receipt_sha256", None)
    resigned["receipt_sha256"] = train_module._canonical_json_sha256(resigned)
    return resigned


def test_validate_training_input_evidence_accepts_bound_positive_fixture(tmp_path):
    features_df, evidence, spec = _direct_training_input_evidence_fixture(tmp_path)

    validated = train_module._validate_training_input_evidence(
        evidence,
        features_df=features_df,
        required_spec=spec,
    )

    assert validated == evidence


def test_validate_training_input_evidence_rejects_bad_receipt_self_hash(tmp_path):
    features_df, evidence, spec = _direct_training_input_evidence_fixture(tmp_path)
    evidence["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="receipt SHA-256 is invalid"):
        train_module._validate_training_input_evidence(
            evidence,
            features_df=features_df,
            required_spec=spec,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("policy_sha256", "policy SHA-256 is stale"),
        ("fullfit_spec_payload_sha256", "spec SHA-256 does not match"),
    ],
)
def test_validate_training_input_evidence_rejects_wrong_policy_or_spec_sha(
    tmp_path,
    field,
    message,
):
    features_df, evidence, spec = _direct_training_input_evidence_fixture(tmp_path)
    evidence[field] = "0" * 64
    evidence = _resign_training_input_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        train_module._validate_training_input_evidence(
            evidence,
            features_df=features_df,
            required_spec=spec,
        )


def test_validate_training_input_evidence_rejects_count_tamper(tmp_path):
    features_df, evidence, spec = _direct_training_input_evidence_fixture(tmp_path)
    evidence["rows_with_verified_t_minus_entry"] = 1
    evidence = _resign_training_input_evidence(evidence)

    with pytest.raises(ValueError, match="verified T-1 row count is stale"):
        train_module._validate_training_input_evidence(
            evidence,
            features_df=features_df,
            required_spec=spec,
        )


def test_validate_training_input_evidence_rejects_unverified_populated_odds(
    tmp_path,
):
    features_df, evidence, spec = _direct_training_input_evidence_fixture(tmp_path)
    features_df.loc[0, "model_odds_verified_prefight"] = False

    with pytest.raises(ValueError, match="do not have verified T-1 provenance"):
        train_module._validate_training_input_evidence(
            evidence,
            features_df=features_df,
            required_spec=spec,
        )
