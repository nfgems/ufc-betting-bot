import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

import src.model.tennis_model as tennis_model


def _training_frame(rows: int, target_pattern: str = "alternating") -> pd.DataFrame:
    if target_pattern == "alternating":
        target = [index % 2 for index in range(rows)]
    elif target_pattern == "blocked":
        split = rows // 2
        target = ([0] * split) + ([1] * (rows - split))
    else:
        raise ValueError(f"Unsupported target pattern: {target_pattern}")

    return pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "tour": np.where(np.arange(rows) % 2 == 0, "atp", "wta"),
            "target": target,
            "diff_elo": np.linspace(-80, 80, rows),
            "diff_surface_elo": np.linspace(-40, 40, rows),
            "best_of_5": np.where(np.arange(rows) % 11 == 0, 1.0, 0.0),
            "a_rank": np.linspace(5, 105, rows),
            "b_rank": np.linspace(75, 175, rows),
            "player_a_rank_points": np.linspace(6000, 1200, rows),
            "player_b_rank_points": np.linspace(3500, 800, rows),
            "a_age": np.linspace(21, 31, rows),
            "b_age": np.linspace(23, 33, rows),
            "diff_age": np.linspace(-2, -2, rows),
            "a_num_matches": np.arange(rows),
            "b_num_matches": np.arange(rows),
        }
    )


def _oos_training_frame() -> pd.DataFrame:
    event_dates = pd.date_range("2021-09-01", "2026-03-10", freq="7D")
    frame = _training_frame(len(event_dates), target_pattern="alternating")
    frame["event_date"] = event_dates
    frame["a_num_matches"] = np.arange(len(event_dates))
    frame["b_num_matches"] = np.arange(len(event_dates))
    return frame


def test_fit_model_uses_sigmoid_calibration_when_temporal_folds_are_valid():
    train_df = _training_frame(240, target_pattern="alternating")

    result = tennis_model._fit_model(
        train_df,
        feature_cols=["diff_elo", "diff_surface_elo", "best_of_5"],
        feature_contract=tennis_model.STAGE1_FEATURE_CONTRACT,
        calibrate=True,
    )

    assert isinstance(result["model"], CalibratedClassifierCV)
    assert result["calibration_method"] == tennis_model.STAGE1_CALIBRATION_METHOD


def test_fit_model_skips_calibration_when_temporal_folds_are_single_class():
    train_df = _training_frame(240, target_pattern="blocked")

    result = tennis_model._fit_model(
        train_df,
        feature_cols=["diff_elo", "diff_surface_elo", "best_of_5"],
        feature_contract=tennis_model.STAGE1_FEATURE_CONTRACT,
        calibrate=True,
    )

    assert isinstance(result["model"], Pipeline)
    assert result["calibration_method"] == "none"


def test_save_and_load_tennis_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tennis_model, "TENNIS_MODELS_DIR", tmp_path)
    train_df = _training_frame(40, target_pattern="alternating")
    model_result = tennis_model._fit_model(
        train_df,
        feature_cols=["diff_elo", "diff_surface_elo", "best_of_5"],
        feature_contract=tennis_model.STAGE1_FEATURE_CONTRACT,
        calibrate=False,
    )

    saved_path = tennis_model.save_tennis_model(model_result, model_name="tennis_test")
    loaded = tennis_model.load_tennis_model(model_name="tennis_test")

    assert saved_path == tmp_path / "tennis_test.pkl"
    assert loaded["feature_cols"] == ["diff_elo", "diff_surface_elo", "best_of_5"]
    assert loaded["feature_contract"] == "stage1_surface_elo_baseline"


def test_prepare_tennis_model_features_derives_stage2_contract_columns():
    frame = _training_frame(5, target_pattern="alternating").drop(columns=["diff_age"])

    prepared = tennis_model.prepare_tennis_model_features(
        frame,
        feature_contract=tennis_model.STAGE2_FEATURE_CONTRACT,
    )

    assert "log_rank_diff" in prepared.columns
    assert "log_rank_points_diff" in prepared.columns
    assert "diff_age" in prepared.columns
    assert prepared.loc[0, "log_rank_diff"] == pytest.approx(
        np.log1p(prepared.loc[0, "b_rank"]) - np.log1p(prepared.loc[0, "a_rank"])
    )


def test_train_tennis_model_fails_when_stage1_feature_column_is_missing(monkeypatch):
    features_df = _training_frame(40, target_pattern="alternating").drop(columns=["best_of_5"])
    monkeypatch.setattr(tennis_model, "save_tennis_model", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Missing required Stage 1 tennis feature columns: best_of_5"):
        tennis_model.train_tennis_model(features_df, model_name="missing_col", min_matches=0)


def test_load_tennis_model_rejects_incomplete_stage1_feature_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(tennis_model, "TENNIS_MODELS_DIR", tmp_path)
    bad_model_path = tmp_path / "bad_contract.pkl"
    tennis_model.joblib.dump(
        {
            "model": tennis_model._build_estimator(),
            "feature_cols": ["diff_elo", "diff_surface_elo"],
            "col_medians": np.array([0.0, 0.0]),
            "feature_contract": "stage1_surface_elo_baseline",
            "calibration_method": "none",
        },
        bad_model_path,
    )

    with pytest.raises(ValueError, match="Invalid Stage 1 tennis feature contract"):
        tennis_model.load_tennis_model(model_name="bad_contract")


def test_train_tennis_model_accepts_minimal_stage1_frame_when_min_history_disabled(monkeypatch):
    features_df = _training_frame(40, target_pattern="alternating")
    features_df["event_date"] = pd.date_range("2022-01-01", periods=40, freq="D")
    monkeypatch.setattr(tennis_model, "save_tennis_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tennis_model,
        "run_walkforward_evaluation",
        lambda *args, **kwargs: {
            "folds": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "metrics": {},
            "summary": {"evaluation_contract": tennis_model.TENNIS_OOS_EVALUATION_CONTRACT},
            "eligible_oos_rows": 0,
        },
    )

    result = tennis_model.train_tennis_model(features_df, model_name="stage1_only", min_matches=0)

    assert result["training_rows"] == 40
    assert result["feature_cols"] == ["diff_elo", "diff_surface_elo", "best_of_5"]
    assert result["evaluation_summary"]["evaluation_contract"] == tennis_model.TENNIS_OOS_EVALUATION_CONTRACT


def test_train_tennis_model_supports_stage2_hybrid_contract(monkeypatch):
    features_df = _training_frame(40, target_pattern="alternating")
    features_df["event_date"] = pd.date_range("2022-01-01", periods=40, freq="D")
    monkeypatch.setattr(tennis_model, "save_tennis_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tennis_model,
        "run_walkforward_evaluation",
        lambda *args, **kwargs: {
            "folds": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "metrics": {},
            "diagnostics": {},
            "summary": {"evaluation_contract": tennis_model.TENNIS_OOS_EVALUATION_CONTRACT},
            "eligible_oos_rows": 0,
        },
    )

    result = tennis_model.train_tennis_model(features_df, model_name="lean_hybrid", min_matches=0)

    assert result["feature_contract"] == tennis_model.STAGE2_FEATURE_CONTRACT
    assert result["feature_cols"] == [
        "diff_elo",
        "diff_surface_elo",
        "best_of_5",
        "log_rank_diff",
        "log_rank_points_diff",
        "diff_age",
    ]
    assert result["oos_artifact_prefix"] == tennis_model.STAGE2_OOS_ARTIFACT_PREFIX


def test_filter_tennis_training_window_enforces_strict_2022_boundary():
    frame = _training_frame(8, target_pattern="alternating")
    frame["event_date"] = pd.to_datetime(
        [
            "2021-12-29",
            "2021-12-31",
            "2022-01-01",
            "2022-01-02",
            "2022-07-01",
            "2023-01-01",
            "2024-12-29",
            "2026-03-10",
        ]
    )

    filtered = tennis_model.filter_tennis_training_window(frame)

    assert filtered["event_date"].min() == pd.Timestamp("2022-01-01")
    assert filtered["event_date"].tolist() == [
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-01-02"),
        pd.Timestamp("2022-07-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2024-12-29"),
        pd.Timestamp("2026-03-10"),
    ]


def test_run_walkforward_evaluation_uses_2023_oos_start_and_covers_latest_eligible_rows():
    features_df = _oos_training_frame()
    prepared = tennis_model._prepare_training_frame(features_df, min_matches=5)
    expected_oos_end = prepared["event_date"].max().normalize() + pd.Timedelta(days=1)

    evaluation = tennis_model.run_walkforward_evaluation(features_df, min_matches=5)
    folds = evaluation["folds"]
    predictions = evaluation["predictions"]
    summary = evaluation["summary"]

    assert folds["start_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-01-01",
        "2023-07-01",
        "2024-01-01",
        "2024-07-01",
        "2025-01-01",
        "2025-07-01",
        "2026-01-01",
    ]
    assert folds["end_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-07-01",
        "2024-01-01",
        "2024-07-01",
        "2025-01-01",
        "2025-07-01",
        "2026-01-01",
        expected_oos_end.strftime("%Y-%m-%d"),
    ]
    assert (predictions["event_date"] >= pd.Timestamp("2023-01-01")).all()
    assert (predictions["event_date"] < expected_oos_end).all()

    eligible = prepared[
        (prepared["event_date"] >= pd.Timestamp("2023-01-01"))
        & (prepared["event_date"] < expected_oos_end)
    ].reset_index(drop=True)

    assert len(predictions) == len(eligible)
    assert predictions["event_date"].tolist() == eligible["event_date"].tolist()
    assert summary["training_start_date"] == "2022-01-01"
    assert summary["training_date_min"] >= "2022-01-01"
    assert summary["first_fold_start_date"] == "2023-01-01"
    assert summary["oos_end_date_exclusive"] == expected_oos_end.strftime("%Y-%m-%d")
    assert summary["eligible_oos_rows"] == len(eligible)
    assert summary["oos_prediction_rows"] == len(predictions)
    assert summary["coverage_matches_eligible_rows"] is True


def test_write_tennis_oos_artifacts_emits_required_files(tmp_path, monkeypatch):
    features_df = _oos_training_frame()
    monkeypatch.setattr(tennis_model, "save_tennis_model", lambda *args, **kwargs: None)

    result = tennis_model.train_tennis_model(features_df, model_name="stage1_oos", min_matches=5)
    artifacts = tennis_model.write_tennis_oos_artifacts(result, tmp_path)

    assert artifacts["folds"].name == "oos_2022plus_2023start_folds.csv"
    assert artifacts["predictions"].name == "oos_2022plus_2023start_predictions.csv"
    assert artifacts["calibration"].name == "oos_2022plus_2023start_calibration.csv"
    assert artifacts["summary"].name == "oos_2022plus_2023start_summary.json"
    for path in artifacts.values():
        assert path.exists()

    folds = pd.read_csv(artifacts["folds"], parse_dates=["start_date", "end_date"])
    predictions = pd.read_csv(
        artifacts["predictions"],
        parse_dates=["event_date", "fold_start_date", "fold_end_date"],
    )
    calibration = pd.read_csv(artifacts["calibration"])
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))

    assert folds.loc[0, "start_date"] == pd.Timestamp("2023-01-01")
    assert (predictions["event_date"] >= pd.Timestamp("2023-01-01")).all()
    assert len(calibration) > 0
    assert summary["evaluation_contract"] == tennis_model.TENNIS_OOS_EVALUATION_CONTRACT
    assert summary["coverage_matches_eligible_rows"] is True
    assert summary["oos_prediction_rows"] == len(predictions)
    assert summary["eligible_oos_rows"] == len(predictions)
    assert result["training_start_date"] == "2022-01-01"
    assert result["training_date_min"] >= "2022-01-01"


def test_run_lockbox_evaluation_uses_strict_holdout_boundary():
    features_df = _oos_training_frame()

    evaluation = tennis_model.run_lockbox_evaluation(
        features_df,
        lockbox_start_date="2026-01-01",
        min_matches=5,
        feature_contract=tennis_model.STAGE2_FEATURE_CONTRACT,
        model_name="lean_hybrid",
    )

    predictions = evaluation["predictions"]
    summary = evaluation["summary"]

    assert (predictions["event_date"] >= pd.Timestamp("2026-01-01")).all()
    assert summary["evaluation_contract"] == tennis_model.TENNIS_LOCKBOX_EVALUATION_CONTRACT
    assert summary["feature_contract"] == tennis_model.STAGE2_FEATURE_CONTRACT
    assert summary["model_name"] == "lean_hybrid"
    assert summary["lockbox_start_date"] == "2026-01-01"
    assert summary["lockbox_rows"] == len(predictions)
    assert summary["train_rows"] > 0


def test_write_tennis_lockbox_artifacts_emits_required_files(tmp_path):
    features_df = _oos_training_frame()
    evaluation = tennis_model.run_lockbox_evaluation(
        features_df,
        lockbox_start_date="2026-01-01",
        min_matches=5,
        feature_contract=tennis_model.STAGE2_FEATURE_CONTRACT,
        model_name="lean_hybrid",
    )

    artifacts = tennis_model.write_tennis_lockbox_artifacts(
        evaluation,
        tmp_path,
        model_name="lean_hybrid",
        lockbox_start_date="2026-01-01",
    )

    assert artifacts["predictions"].name == "lockbox_20260101_lean_hybrid_predictions.csv"
    assert artifacts["calibration"].name == "lockbox_20260101_lean_hybrid_calibration.csv"
    assert artifacts["summary"].name == "lockbox_20260101_lean_hybrid_summary.json"
    for path in artifacts.values():
        assert path.exists()
