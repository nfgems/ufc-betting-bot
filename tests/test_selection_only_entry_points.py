import pickle
import sys

import pandas as pd
import pytest

import scripts.e5_conviction_sweep as e5_sweep
import scripts.e8_selection_bias as e8_sweep
import scripts.e9_logit_stack as e9_stack
import scripts.optuna_tune_v6 as optuna_v6
from src.strategy import duo_trader_sweep
from src.strategy import finalist_sweep
from src.strategy import model_lab
from src.strategy import run_evaluation as run_eval
from src.strategy import triple_trader_backtest as triple_backtest
from src.strategy.duo_trader_sweep import (
    _selection_prediction_cache_payload,
    _selection_predictions_from_cache_payload,
)
from src.strategy.model_variants import VariantConfig


def _dated_fold(fold_id: int, date: str) -> tuple[int, pd.DataFrame]:
    return (
        fold_id,
        pd.DataFrame(
            {
                "event_date": pd.to_datetime([date]),
                "fighter_a": [f"A{fold_id}"],
                "fighter_b": [f"B{fold_id}"],
                "target": [fold_id % 2],
                "prob_a": [0.6],
                "prob_b": [0.4],
                "fold": [fold_id],
            }
        ),
    )


def _long_features_frame() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2026-08-01", freq="7D")
    return pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A{i}" for i in range(len(dates))],
            "fighter_b": [f"B{i}" for i in range(len(dates))],
            "target": [i % 2 for i in range(len(dates))],
            "diff_stat": [0.1] * len(dates),
            "a_num_fights": [3] * len(dates),
            "b_num_fights": [3] * len(dates),
        }
    )


def test_all_fold_generation_requires_an_explicit_override():
    with pytest.raises(ValueError, match="all-fold model experiments are disabled"):
        model_lab.generate_variant_fold_predictions(
            pd.DataFrame(),
            VariantConfig(name="guard", description="guard"),
            evaluation_partition="all",
        )


def test_stage1_worker_rejects_a_missing_confirmation_reserve_contract():
    with pytest.raises(ValueError, match="missing the reserved_confirmation_folds"):
        run_eval._evaluate_cell_worker({})


def test_triple_trader_sweep_never_fits_or_scores_reserved_folds(monkeypatch):
    features_df = _long_features_frame()
    fitted_training_ends: list[pd.Timestamp] = []
    scored_dates: list[pd.Timestamp] = []
    preflight_manifests: list[list[tuple[int, pd.DataFrame]]] = []

    monkeypatch.setattr(triple_backtest, "get_feature_columns", lambda _df: ["diff_stat"])
    monkeypatch.setattr(
        triple_backtest,
        "get_feature_columns_no_odds",
        lambda _df: ["diff_stat"],
    )

    def fake_train(train_df, _feature_cols, _variant):
        fitted_training_ends.append(pd.to_datetime(train_df["event_date"]).max())
        return {}

    def fake_predict(test_df, _model_result):
        scored_dates.extend(pd.to_datetime(test_df["event_date"]).tolist())
        return test_df[
            ["event_date", "fighter_a", "fighter_b", "target"]
        ].assign(prob_a=0.6, prob_b=0.4)

    monkeypatch.setattr(triple_backtest, "train_variant_model", fake_train)
    monkeypatch.setattr(triple_backtest, "_predict_batch_with_model", fake_predict)
    monkeypatch.setattr(
        triple_backtest,
        "_preflight_selection_manifest",
        lambda manifest: preflight_manifests.append(manifest),
    )
    monkeypatch.setattr(triple_backtest, "_merge_historical_odds", lambda frame: frame)
    monkeypatch.setattr(
        triple_backtest,
        "_resolve_market_odds",
        lambda frame, *_args: (frame, {}),
    )

    fold_predictions = triple_backtest._generate_walk_forward_predictions(
        features_df,
        retrain_months=6,
        initial_train_years=5,
        bet_start_date="2022-01-01",
    )

    dates = pd.to_datetime(features_df["event_date"])
    full_windows = triple_backtest._walk_forward_fold_windows(
        dates,
        initial_train_years=5,
        retrain_months=6,
        bet_start_date="2022-01-01",
    )
    selection_windows = full_windows[:-model_lab.DEFAULT_CONFIRMATION_FOLD_COUNT]
    confirmation_start = full_windows[-model_lab.DEFAULT_CONFIRMATION_FOLD_COUNT][1]

    assert [fold_id for fold_id, _frame in fold_predictions] == [
        window[0] for window in selection_windows
    ]
    assert len(preflight_manifests) == 1
    assert [fold_id for fold_id, _frame in preflight_manifests[0]] == [
        window[0] for window in full_windows
    ]
    assert len(fitted_training_ends) == 2 * len(selection_windows)
    assert scored_dates
    assert max(scored_dates) < confirmation_start


@pytest.mark.parametrize(
    "entry_point",
    [
        triple_backtest.run_triple_trader_backtest,
        triple_backtest.run_priority_trader_backtest,
    ],
)
def test_legacy_triple_modes_use_the_selection_fold_guard(
    monkeypatch,
    entry_point,
):
    captured = {}

    def stop_after_capture(features_df, **kwargs):
        captured["features_df"] = features_df
        captured.update(kwargs)
        raise RuntimeError("captured selection guard")

    monkeypatch.setattr(
        triple_backtest,
        "get_feature_columns",
        lambda _frame: ["diff_stat"],
    )
    monkeypatch.setattr(
        triple_backtest,
        "get_feature_columns_no_odds",
        lambda _frame: ["diff_stat"],
    )
    monkeypatch.setattr(
        triple_backtest,
        "_selection_fold_windows",
        stop_after_capture,
    )

    with pytest.raises(RuntimeError, match="captured selection guard"):
        entry_point(_long_features_frame())

    assert captured["bet_start_date"] == "2022-01-01"
    assert captured["initial_train_years"] == 5
    assert captured["retrain_months"] == 6


def test_optuna_outer_holdout_stops_before_confirmation_reserve(monkeypatch):
    features_df = _long_features_frame()
    confirmation_start = optuna_v6._confirmation_reserve_start(features_df)
    captured: dict[str, pd.Timestamp] = {}
    preflight_frames: list[pd.DataFrame] = []
    variant = optuna_v6._build_variant_from_params(
        dict(optuna_v6.BASELINE_PARAMS),
        name="selection_only",
        description="selection only",
    )
    variant.feature_cols = ["diff_stat"]

    def fake_train(train_df, feature_cols, **_kwargs):
        captured["train_max"] = pd.to_datetime(train_df["event_date"]).max()
        return {
            "model": object(),
            "feature_cols": list(feature_cols),
            "col_medians": [0.0],
        }

    def fake_predict(holdout_df, **_kwargs):
        captured["score_max"] = pd.to_datetime(holdout_df["event_date"]).max()
        return holdout_df.assign(prob_a=0.6, prob_b=0.4)

    monkeypatch.setattr(optuna_v6, "train_xgboost", fake_train)
    monkeypatch.setattr(optuna_v6, "predict_batch", fake_predict)
    monkeypatch.setattr(
        optuna_v6,
        "_preflight_scored_selection_rows",
        lambda frame: preflight_frames.append(frame.copy()),
    )

    predictions = optuna_v6._train_then_predict_outer_holdout(features_df, variant)

    assert captured["train_max"] < optuna_v6.OUTER_HOLDOUT_START
    assert captured["score_max"] < confirmation_start
    assert len(preflight_frames) == 1
    assert pd.to_datetime(preflight_frames[0]["event_date"]).max() < confirmation_start
    assert pd.to_datetime(predictions["event_date"]).max() < confirmation_start
    assert predictions.attrs["outer_holdout_training"][
        "holdout_end_date_exclusive"
    ] == confirmation_start.date().isoformat()


def test_selection_prediction_cache_rejects_legacy_and_reserved_fold_rows():
    selection = [_dated_fold(1, "2023-01-01"), _dated_fold(2, "2024-01-01")]
    manifest = selection + [
        _dated_fold(3, "2025-01-01"),
        _dated_fold(4, "2026-01-01"),
    ]
    payload = _selection_prediction_cache_payload(selection, manifest)

    assert _selection_predictions_from_cache_payload(payload) == selection
    with pytest.raises(ValueError, match="lacks selection-partition provenance"):
        _selection_predictions_from_cache_payload(manifest)

    payload["fold_predictions"] = selection + [_dated_fold(3, "2025-01-01")]
    with pytest.raises(ValueError, match="exposes a reserved confirmation fold"):
        _selection_predictions_from_cache_payload(payload)


@pytest.mark.parametrize("script_module", [e5_sweep, e8_sweep, e9_stack])
def test_legacy_raw_prediction_cache_is_rejected_by_sweep_entry_point(
    tmp_path,
    monkeypatch,
    script_module,
):
    cache_path = tmp_path / "legacy-all-fold.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump([_dated_fold(9, "2025-10-01")], handle)
    monkeypatch.setattr(
        sys,
        "argv",
        [script_module.__name__, "--preds-cache", str(cache_path)],
    )

    with pytest.raises(ValueError, match="selection-partition provenance"):
        script_module.main()


@pytest.mark.parametrize("script_module", [e5_sweep, e8_sweep, e9_stack])
def test_valid_selection_cache_is_preflighted_before_sweep_scoring(
    tmp_path,
    monkeypatch,
    script_module,
):
    selection = [_dated_fold(1, "2023-01-01"), _dated_fold(2, "2024-01-01")]
    payload = _selection_prediction_cache_payload(
        selection,
        selection + [_dated_fold(3, "2025-01-01"), _dated_fold(4, "2026-01-01")],
    )
    cache_path = tmp_path / "selection-v1.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)
    captured = {}

    def stop_after_capture(fold_predictions):
        captured["fold_predictions"] = fold_predictions
        raise RuntimeError("captured selection preflight")

    monkeypatch.setattr(
        script_module,
        "_preflight_cached_selection_predictions",
        stop_after_capture,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [script_module.__name__, "--preds-cache", str(cache_path)],
    )

    with pytest.raises(RuntimeError, match="captured selection preflight"):
        script_module.main()

    assert [fold_id for fold_id, _frame in captured["fold_predictions"]] == [1, 2]


@pytest.mark.parametrize("script_module", [e5_sweep, e8_sweep])
def test_fresh_sweep_generation_explicitly_requests_selection_partition(
    tmp_path,
    monkeypatch,
    script_module,
):
    captured = {}

    def stop_after_capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured generation contract")

    monkeypatch.setattr(
        script_module,
        "_generate_walk_forward_predictions",
        stop_after_capture,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script_module.__name__,
            "--preds-cache",
            str(tmp_path / "new-cache.pkl"),
        ],
    )

    with pytest.raises(RuntimeError, match="captured generation contract"):
        script_module.main()

    assert captured["evaluation_partition"] == "selection"
    assert captured["confirmation_fold_count"] == 2
    assert captured["return_fold_manifest"] is True
    assert captured["allow_all_folds"] is False


def test_duo_sweep_callers_explicitly_request_selection_partition(monkeypatch):
    captured: list[dict] = []

    def stop_after_capture(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("captured generation contract")

    monkeypatch.setattr(
        duo_trader_sweep,
        "_generate_walk_forward_predictions",
        stop_after_capture,
    )
    with pytest.raises(RuntimeError, match="captured generation contract"):
        duo_trader_sweep.run_sweep_backtest(variant_name="baseline")
    with pytest.raises(RuntimeError, match="captured generation contract"):
        duo_trader_sweep.run_betsapi_challenger_sweep_comparison_with_configs(
            [
                duo_trader_sweep.SweepConfig(
                    name="challenger",
                    variant_name="baseline",
                )
            ],
            include_production_baseline=False,
        )

    assert len(captured) == 2
    assert all(call["evaluation_partition"] == "selection" for call in captured)
    assert all(call["confirmation_fold_count"] == 2 for call in captured)
    assert all(call["allow_all_folds"] is False for call in captured)


def test_finalist_sweep_explicitly_requests_selection_partition(monkeypatch):
    captured = {}

    def stop_after_capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured generation contract")

    monkeypatch.setattr(
        finalist_sweep,
        "_generate_walk_forward_predictions",
        stop_after_capture,
    )

    with pytest.raises(RuntimeError, match="captured generation contract"):
        finalist_sweep.run_finalist_sweep(["baseline"], run_narrow=False)

    assert captured["evaluation_partition"] == "selection"
    assert captured["confirmation_fold_count"] == 2
    assert captured["allow_all_folds"] is False
