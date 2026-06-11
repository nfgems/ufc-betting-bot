import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

import src.strategy.backtest as backtest
import src.model.compare as compare_module
import src.model.train as train_module
import src.model.training_spec as training_spec_module


def _single_fight_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Fighter A",
            "fighter_b": "Fighter B",
            "target": 1,
            "a_fair_prob_avg": 0.50,
            "b_fair_prob_avg": 0.50,
            "a_num_fights": 5,
            "b_num_fights": 5,
        }
    ])


def test_signal_only_strategy_skips_agreement_model_load(monkeypatch):
    calls: list[str] = []

    def fake_load_model(model_name: str):
        calls.append(model_name)
        return {
            "model": object(),
            "feature_cols": ["dummy"],
            "col_medians": np.array([0.0]),
        }

    def fake_predict_batch(features_df, model_name="xgboost", model_result=None):
        result = features_df.copy()
        result["prob_a"] = 0.70
        result["prob_b"] = 0.30
        result["predicted_winner"] = "a"
        result["confidence"] = 0.70
        return result

    monkeypatch.setattr(backtest, "load_model", fake_load_model)
    monkeypatch.setattr(backtest, "predict_batch", fake_predict_batch)

    result = backtest.run_backtest(
        _single_fight_df(),
        model_name="xgboost",
        model_result={
            "model": object(),
            "feature_cols": ["dummy"],
            "col_medians": np.array([0.0]),
        },
        strategy_config=backtest.ODDS_AWARE_SIGNAL_ONLY_STRATEGY,
        use_historical_odds=False,
        min_edge=0.01,
    )

    assert calls == []
    assert len(result["bet_log"]) == 1


def test_no_odds_signal_only_uses_raw_primary_probabilities():
    predictions = pd.DataFrame([
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Fighter A",
            "fighter_b": "Fighter B",
            "target": 1,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_no_odds_prob_a": 0.70,
            "xgboost_no_odds_prob_b": 0.30,
        }
    ])

    result = backtest._simulate_backtest_predictions(
        predictions,
        backtest.NO_ODDS_SIGNAL_ONLY_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
    )

    bet_log = result["bet_log"]
    assert len(bet_log) == 1
    assert bet_log.loc[0, "model_prob"] == pytest.approx(0.70)
    assert bet_log.loc[0, "blended_prob"] == pytest.approx(0.70)
    assert bet_log.loc[0, "blend_weight_used"] == pytest.approx(1.0)
    assert pd.isna(bet_log.loc[0, "agreement_prob"])


def test_production_gated_requires_no_odds_agreement():
    weak_agreement = pd.DataFrame([
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Fighter A",
            "fighter_b": "Fighter B",
            "target": 1,
            "a_market_prob": 0.55,
            "b_market_prob": 0.45,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_prob_a": 0.78,
            "xgboost_prob_b": 0.22,
            "xgboost_no_odds_prob_a": 0.55,
            "xgboost_no_odds_prob_b": 0.45,
        }
    ])
    weak_result = backtest._simulate_backtest_predictions(
        weak_agreement,
        backtest.PRODUCTION_GATED_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
    )
    assert weak_result["bet_log"].empty

    strong_agreement = weak_agreement.copy()
    strong_agreement.loc[0, "xgboost_no_odds_prob_a"] = 0.70
    strong_agreement.loc[0, "xgboost_no_odds_prob_b"] = 0.30
    strong_result = backtest._simulate_backtest_predictions(
        strong_agreement,
        backtest.PRODUCTION_GATED_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
    )
    assert len(strong_result["bet_log"]) == 1


def test_production_gated_can_override_max_decimal_odds():
    predictions = pd.DataFrame([
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Fighter A",
            "fighter_b": "Fighter B",
            "target": 1,
            "a_market_prob": 1 / 3.5,
            "b_market_prob": 1 - (1 / 3.5),
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_prob_a": 0.62,
            "xgboost_prob_b": 0.38,
            "xgboost_no_odds_prob_a": 0.60,
            "xgboost_no_odds_prob_b": 0.40,
        }
    ])

    default_result = backtest._simulate_backtest_predictions(
        predictions,
        backtest.PRODUCTION_GATED_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
    )
    assert default_result["bet_log"].empty

    looser_cap_strategy = replace(backtest.PRODUCTION_GATED_STRATEGY, max_decimal_odds=4.0)
    looser_cap_result = backtest._simulate_backtest_predictions(
        predictions,
        looser_cap_strategy,
        initial_bankroll=100.0,
        min_edge=0.01,
    )
    assert len(looser_cap_result["bet_log"]) == 1


def test_walkforward_comparison_smoke(monkeypatch):
    spec = training_spec_module.NamedModelTrainingSpec(
        name="smoke_walkforward_spec",
        feature_cols=["diff_stat", "a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        impute_strategy="native_nan",
    )

    def fake_train_xgboost(train_df, feature_cols, calibrate=True, **kwargs):
        return {
            "model": object(),
            "feature_cols": list(feature_cols),
            "col_medians": np.zeros(len(feature_cols)),
            "tag": "with_odds" if "a_implied_prob" in feature_cols else "no_odds",
        }

    def fake_predict_batch(features_df, model_name="xgboost", model_result=None):
        result = features_df.copy()
        base = np.where(features_df["diff_stat"] > 0, 0.68, 0.32)
        if model_result.get("tag") == "no_odds":
            base = np.where(features_df["diff_stat"] > 0, 0.62, 0.38)
        result["prob_a"] = base
        result["prob_b"] = 1.0 - base
        result["predicted_winner"] = np.where(base > 0.5, "a", "b")
        result["confidence"] = np.maximum(base, 1.0 - base)
        return result

    def fake_bootstrap_metric_comparison(y_true, probs_baseline, probs_challenger, metric_fn, **kwargs):
        return {
            "baseline": float(metric_fn(y_true, probs_baseline)),
            "challenger": float(metric_fn(y_true, probs_challenger)),
            "diff": float(metric_fn(y_true, probs_challenger) - metric_fn(y_true, probs_baseline)),
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "p_value": 0.5,
        }

    def fake_mcnemar_test(y_true, preds_a, preds_b):
        return {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0}

    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(training_spec_module, "full_live_contract_spec", lambda: spec)
    monkeypatch.setattr(training_spec_module, "materialize_and_validate_spec_features", lambda df, _spec: df)
    monkeypatch.setattr(backtest, "predict_batch", fake_predict_batch)
    monkeypatch.setattr(backtest, "_merge_historical_odds", lambda df, **kwargs: df)
    monkeypatch.setattr(compare_module, "bootstrap_metric_comparison", fake_bootstrap_metric_comparison)
    monkeypatch.setattr(compare_module, "mcnemar_test", fake_mcnemar_test)

    dates = pd.date_range("2020-01-01", periods=220, freq="3D")
    diff_stat = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    features_df = pd.DataFrame({
        "event_date": dates,
        "fighter_a": [f"A{i}" for i in range(len(dates))],
        "fighter_b": [f"B{i}" for i in range(len(dates))],
        "target": (diff_stat > 0).astype(int),
        "diff_stat": diff_stat,
        "a_implied_prob": 0.55,
        "b_implied_prob": 0.45,
        "diff_implied_prob": 0.10,
        "a_fair_prob_avg": 0.50,
        "b_fair_prob_avg": 0.50,
        "a_num_fights": 5,
        "b_num_fights": 5,
    })

    result = backtest.run_walkforward_strategy_comparison(
        features_df,
        retrain_months=6,
        initial_train_years=1,
        initial_bankroll=100.0,
        min_edge=0.01,
        bet_start_date="2020-01-01",
        write_artifacts=False,
    )

    summary = result["summary"]
    fold_results = result["fold_results"]
    predictive_metrics = result["predictive_metrics"]

    assert {"strategy", "primary_model", "roi", "total_profit"}.issubset(summary.columns)
    assert set(summary["strategy"]) == {
        "odds_aware_signal_only",
        "no_odds_signal_only",
        "production_gated",
    }
    assert {"strategy", "fold", "test_fights", "bets", "roi"}.issubset(fold_results.columns)
    assert set(predictive_metrics["models"]) == {"xgboost", "xgboost_no_odds"}


def test_walkforward_comparison_uses_promoted_spec_materialization(monkeypatch):
    spec = training_spec_module.NamedModelTrainingSpec(
        name="walkforward_contract_test",
        feature_cols=["core_feature", "is_rematch", "a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        impute_strategy="native_nan",
        xgb_params={"n_estimators": 11, "max_depth": 2},
        calibration_method="sigmoid",
        calibration_cv="random_5fold",
        add_rematch_features=True,
    )
    train_calls: list[dict] = []

    def fake_materialize(features_df, incoming_spec):
        assert incoming_spec is spec
        return features_df.assign(is_rematch=np.where(features_df["core_feature"] > 0, 1, 0))

    def fake_train_xgboost(train_df, feature_cols, calibrate=True, **kwargs):
        train_calls.append(
            {
                "train_df": train_df.copy(),
                "feature_cols": list(feature_cols),
                "kwargs": dict(kwargs),
            }
        )
        return {
            "model": object(),
            "feature_cols": list(feature_cols),
            "col_medians": np.zeros(len(feature_cols)),
            "tag": "with_odds" if "a_implied_prob" in feature_cols else "no_odds",
        }

    def fake_predict_batch(features_df, model_name="xgboost", model_result=None):
        result = features_df.copy()
        base = np.where(features_df["core_feature"] > 0, 0.67, 0.33)
        if model_result.get("tag") == "no_odds":
            base = np.where(features_df["core_feature"] > 0, 0.61, 0.39)
        result["prob_a"] = base
        result["prob_b"] = 1.0 - base
        result["predicted_winner"] = np.where(base > 0.5, "a", "b")
        result["confidence"] = np.maximum(base, 1.0 - base)
        return result

    def fake_bootstrap_metric_comparison(y_true, probs_baseline, probs_challenger, metric_fn, **kwargs):
        return {
            "baseline": float(metric_fn(y_true, probs_baseline)),
            "challenger": float(metric_fn(y_true, probs_challenger)),
            "diff": float(metric_fn(y_true, probs_challenger) - metric_fn(y_true, probs_baseline)),
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "p_value": 0.5,
        }

    monkeypatch.setattr(training_spec_module, "materialize_and_validate_spec_features", fake_materialize)
    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(backtest, "predict_batch", fake_predict_batch)
    monkeypatch.setattr(backtest, "_merge_historical_odds", lambda df, **kwargs: df)
    monkeypatch.setattr(compare_module, "bootstrap_metric_comparison", fake_bootstrap_metric_comparison)
    monkeypatch.setattr(compare_module, "mcnemar_test", lambda *_args, **_kwargs: {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0})

    dates = pd.date_range("2020-01-01", periods=220, freq="3D")
    core_feature = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    features_df = pd.DataFrame({
        "event_date": dates,
        "fighter_a": [f"A{i}" for i in range(len(dates))],
        "fighter_b": [f"B{i}" for i in range(len(dates))],
        "target": (core_feature > 0).astype(int),
        "core_feature": core_feature,
        "a_implied_prob": 0.55,
        "b_implied_prob": 0.45,
        "diff_implied_prob": 0.10,
        "a_fair_prob_avg": 0.50,
        "b_fair_prob_avg": 0.50,
        "a_num_fights": 5,
        "b_num_fights": 5,
    })

    result = backtest.run_walkforward_strategy_comparison(
        features_df,
        retrain_months=6,
        initial_train_years=1,
        initial_bankroll=100.0,
        min_edge=0.01,
        bet_start_date="2020-01-01",
        write_artifacts=False,
        spec=spec,
    )

    assert train_calls
    assert train_calls[0]["feature_cols"] == spec.feature_cols
    assert train_calls[0]["kwargs"]["impute_strategy"] == "native_nan"
    assert train_calls[0]["kwargs"]["xgb_params"] == {"n_estimators": 11, "max_depth": 2}
    assert train_calls[0]["kwargs"]["calibration_method"] == "sigmoid"
    assert train_calls[0]["kwargs"]["calibration_cv"] == "random_5fold"
    assert train_calls[1]["feature_cols"] == ["core_feature", "is_rematch"]
    assert result["spec"] is spec
