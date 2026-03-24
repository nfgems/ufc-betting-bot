from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import train as train_module
from src.model.training_spec import NamedModelTrainingSpec, compute_feature_family_coverage


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
