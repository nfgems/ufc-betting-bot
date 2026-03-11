import numpy as np
import pandas as pd

import src.strategy.model_lab as model_lab
import src.strategy.model_variants as model_variants
import src.model.train as train_module


def test_baseline_variant_matches_current_production_defaults():
    baseline = model_variants.baseline()
    params = model_variants._production_xgb_params()

    assert baseline.calibration_method == "isotonic"
    assert baseline.calibration_cv == "timeseries_5fold"
    assert baseline.impute_with_indicators is True
    assert params == {
        "n_estimators": 135,
        "max_depth": 7,
        "learning_rate": 0.0124,
        "subsample": 0.659,
        "colsample_bytree": 0.706,
        "min_child_weight": 6,
        "gamma": 0.444,
        "reg_alpha": 0.00443,
        "reg_lambda": 0.00772,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }


def test_run_variant_walkforward_passes_agreement_threshold_override(monkeypatch):
    captured = []

    def fake_train_variant_model(train_df, feature_cols, variant):
        return {"tag": "no_odds" if variant.name == "_no_odds" else "primary"}

    def fake_predict_batch_with_model(features_df, model_result):
        result = features_df.copy()
        prob_a = 0.62 if model_result["tag"] == "no_odds" else 0.68
        result["prob_a"] = prob_a
        result["prob_b"] = 1.0 - prob_a
        return result

    def fake_passes_filters(*args, **kwargs):
        captured.append(kwargs)
        return True

    monkeypatch.setattr(model_lab, "train_variant_model", fake_train_variant_model)
    monkeypatch.setattr(model_lab, "_predict_batch_with_model", fake_predict_batch_with_model)
    monkeypatch.setattr(model_lab, "_merge_historical_odds", lambda df: df)
    monkeypatch.setattr(model_lab, "_passes_filters", fake_passes_filters)

    dates = pd.date_range("2020-01-01", periods=160, freq="7D")
    features_df = pd.DataFrame({
        "event_date": dates,
        "fighter_a": [f"A{i}" for i in range(len(dates))],
        "fighter_b": [f"B{i}" for i in range(len(dates))],
        "target": np.ones(len(dates), dtype=int),
        "a_fair_prob_avg": 0.50,
        "b_fair_prob_avg": 0.50,
        "a_num_fights": 5,
        "b_num_fights": 5,
        "diff_stat": 1.0,
    })

    result = model_lab.run_variant_walkforward(
        features_df,
        model_variants.stronger_gate_020(),
        retrain_months=6,
        initial_train_years=1,
        initial_bankroll=100.0,
        bet_start_date="2020-01-01",
    )

    assert not result["bet_log"].empty
    assert captured
    assert all(call["model_agreement_min_edge"] == 0.02 for call in captured)


def test_train_variant_model_uses_variant_time_decay_half_life(monkeypatch):
    observed = []

    class DummyXGB:
        def __init__(self, **kwargs):
            self.feature_importances_ = np.array([1.0])

        def fit(self, X, y, sample_weight=None):
            return self

    def fake_compute_sample_weights(train_df):
        observed.append(train_module.TIME_DECAY_HALF_LIFE_DAYS)
        return np.ones(len(train_df))

    monkeypatch.setattr(model_variants, "XGBClassifier", DummyXGB)
    monkeypatch.setattr(model_variants, "_compute_sample_weights", fake_compute_sample_weights)

    train_df = pd.DataFrame({
        "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "target": [1, 0, 1],
        "feature": [0.1, 0.2, 0.3],
    })
    variant = model_variants.shorter_decay()
    variant.calibration_method = "none"

    original_half_life = train_module.TIME_DECAY_HALF_LIFE_DAYS
    model_variants.train_variant_model(train_df, ["feature"], variant)

    assert observed == [365]
    assert train_module.TIME_DECAY_HALF_LIFE_DAYS == original_half_life
