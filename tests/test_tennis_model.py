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
            "target": target,
            "diff_elo": np.linspace(-80, 80, rows),
            "diff_surface_elo": np.linspace(-40, 40, rows),
            "best_of_5": np.where(np.arange(rows) % 11 == 0, 1.0, 0.0),
        }
    )


def test_fit_model_uses_sigmoid_calibration_when_temporal_folds_are_valid():
    train_df = _training_frame(240, target_pattern="alternating")

    result = tennis_model._fit_model(
        train_df,
        feature_cols=["diff_elo", "diff_surface_elo", "best_of_5"],
        calibrate=True,
    )

    assert isinstance(result["model"], CalibratedClassifierCV)
    assert result["calibration_method"] == tennis_model.STAGE1_CALIBRATION_METHOD


def test_fit_model_skips_calibration_when_temporal_folds_are_single_class():
    train_df = _training_frame(240, target_pattern="blocked")

    result = tennis_model._fit_model(
        train_df,
        feature_cols=["diff_elo", "diff_surface_elo", "best_of_5"],
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
        calibrate=False,
    )

    saved_path = tennis_model.save_tennis_model(model_result, model_name="tennis_test")
    loaded = tennis_model.load_tennis_model(model_name="tennis_test")

    assert saved_path == tmp_path / "tennis_test.pkl"
    assert loaded["feature_cols"] == ["diff_elo", "diff_surface_elo", "best_of_5"]
    assert loaded["feature_contract"] == "stage1_surface_elo_baseline"


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
    monkeypatch.setattr(tennis_model, "save_tennis_model", lambda *args, **kwargs: None)

    result = tennis_model.train_tennis_model(features_df, model_name="stage1_only", min_matches=0)

    assert result["training_rows"] == 40
    assert result["feature_cols"] == ["diff_elo", "diff_surface_elo", "best_of_5"]
