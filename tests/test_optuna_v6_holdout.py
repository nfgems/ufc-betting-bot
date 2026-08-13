import pandas as pd

import scripts.optuna_tune_v6 as optuna_v6


class _FakeTrial:
    number = 7

    def __init__(self):
        self.params = {}
        self.reported = []

    def suggest_int(self, name, low, high, step=None):
        value = int(optuna_v6.BASELINE_PARAMS[name])
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None, log=False):
        value = float(optuna_v6.BASELINE_PARAMS[name])
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = optuna_v6.BASELINE_PARAMS[name]
        self.params[name] = value
        return value

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return False


def test_split_selection_and_holdout_frames_separates_outer_holdout():
    features_df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2024-12-31", "2025-01-01", "2025-03-01"]),
            "target": [1, 0, 1],
            "feature": [0.1, 0.2, 0.3],
        }
    )

    selection_df, holdout_df = optuna_v6.split_selection_and_holdout_frames(features_df)

    assert selection_df["event_date"].max() < optuna_v6.OUTER_HOLDOUT_START
    assert holdout_df["event_date"].min() >= optuna_v6.OUTER_HOLDOUT_START


def test_objective_receives_only_pre_holdout_selection_frame(monkeypatch):
    full_features_df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2024-12-31", "2025-01-01"]),
            "target": [1, 0],
            "feature": [0.1, 0.2],
        }
    )
    selection_df, _holdout_df = optuna_v6.split_selection_and_holdout_frames(full_features_df)
    trial = _FakeTrial()

    def fake_generate_variant_fold_predictions(features_df, variant, **kwargs):
        assert features_df["event_date"].max() < optuna_v6.OUTER_HOLDOUT_START
        prediction_dates = pd.date_range("2022-01-01", periods=120, freq="7D")
        return [
            (
                1,
                pd.DataFrame(
                    {
                        "event_date": prediction_dates,
                        "target": [1, 0] * 60,
                        "prob_a": [0.7, 0.4] * 60,
                    }
                ),
            )
        ]

    monkeypatch.setattr(optuna_v6, "generate_variant_fold_predictions", fake_generate_variant_fold_predictions)

    score = optuna_v6.objective(trial, selection_df)

    assert score < float("inf")
    assert trial.reported


def test_outer_holdout_evaluation_trains_only_on_pre_2025_rows(monkeypatch):
    pre_holdout_dates = pd.date_range("2024-01-01", periods=120, freq="D")
    holdout_dates = pd.to_datetime(["2025-01-10", "2025-02-10"])
    features_df = pd.DataFrame(
        {
            "event_date": pre_holdout_dates.tolist() + holdout_dates.tolist(),
            "target": ([1, 0] * 60) + [0, 1],
            "fighter_a": [f"A{i}" for i in range(122)],
            "fighter_b": [f"B{i}" for i in range(122)],
            "diff_stat": [0.1] * 122,
            "a_num_fights": [5] * 122,
            "b_num_fights": [5] * 122,
        }
    )
    captured = {}
    variant = optuna_v6._build_variant_from_params(
        dict(optuna_v6.BASELINE_PARAMS),
        name="baseline",
        description="baseline",
    )
    variant.feature_cols = ["diff_stat"]

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        captured["train_max_date"] = train_df["event_date"].max()
        captured["feature_cols"] = list(feature_cols)
        return {"model": object(), "feature_cols": list(feature_cols), "col_medians": [0.0]}

    def fake_predict_batch(holdout_df, model_name, model_result):
        captured["holdout_min_date"] = holdout_df["event_date"].min()
        return holdout_df.assign(prob_a=0.6, prob_b=0.4)

    monkeypatch.setattr(optuna_v6, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(optuna_v6, "predict_batch", fake_predict_batch)

    predictions = optuna_v6._train_then_predict_outer_holdout(features_df, variant)

    assert captured["train_max_date"] < optuna_v6.OUTER_HOLDOUT_START
    assert captured["holdout_min_date"] >= optuna_v6.OUTER_HOLDOUT_START
    assert predictions["event_date"].min() >= optuna_v6.OUTER_HOLDOUT_START


def test_outer_holdout_evaluation_falls_back_when_timeseries_calibration_degenerates(monkeypatch):
    pre_holdout_dates = pd.date_range("2024-01-01", periods=120, freq="D")
    holdout_dates = pd.to_datetime(["2025-01-10", "2025-02-10"])
    features_df = pd.DataFrame(
        {
            "event_date": pre_holdout_dates.tolist() + holdout_dates.tolist(),
            "target": ([1, 0] * 60) + [0, 1],
            "fighter_a": [f"A{i}" for i in range(122)],
            "fighter_b": [f"B{i}" for i in range(122)],
            "diff_stat": [0.1] * 122,
            "a_num_fights": [5] * 122,
            "b_num_fights": [5] * 122,
        }
    )
    variant = optuna_v6._build_variant_from_params(
        dict(optuna_v6.BASELINE_PARAMS),
        name="baseline",
        description="baseline",
    )
    variant.feature_cols = ["diff_stat"]
    calls = []

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        calls.append(
            {
                "train_max_date": train_df["event_date"].max(),
                "calibration_cv": kwargs.get("calibration_cv"),
                "calibrate": kwargs.get("calibrate"),
            }
        )
        if kwargs.get("calibration_cv") == "timeseries_5fold":
            raise ValueError("Invalid classes inferred from unique values of `y`.  Expected: [0], got [1.]")
        return {"model": object(), "feature_cols": list(feature_cols), "col_medians": [0.0]}

    def fake_predict_batch(holdout_df, model_name, model_result):
        return holdout_df.assign(prob_a=0.6, prob_b=0.4)

    monkeypatch.setattr(optuna_v6, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(optuna_v6, "predict_batch", fake_predict_batch)

    predictions = optuna_v6._train_then_predict_outer_holdout(features_df, variant)

    assert len(calls) >= 2
    assert calls[0]["calibration_cv"] == "timeseries_5fold"
    assert calls[1]["calibration_cv"] == "temporal_holdout"
    assert all(call["train_max_date"] < optuna_v6.OUTER_HOLDOUT_START for call in calls)
    assert predictions.attrs["outer_holdout_training"]["fallback_used"] is True
    assert predictions.attrs["outer_holdout_training"]["effective_calibration_cv"] == "temporal_holdout"
