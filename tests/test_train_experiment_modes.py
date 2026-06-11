"""E2/E11/E12: refit calibration, weighted calibration, antithetic noise."""

import numpy as np
import pandas as pd
import pytest

from src.model.train import (
    HoldoutCalibratedRefitModel,
    _add_antithetic_odds_noise,
    train_xgboost,
)


def _synthetic_training_frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    a_skill = rng.normal(0, 1, n)
    b_skill = rng.normal(0, 1, n)
    prob = 1.0 / (1.0 + np.exp(-(a_skill - b_skill)))
    target = (rng.uniform(size=n) < prob).astype(int)
    a_prob = np.clip(prob + rng.normal(0, 0.05, n), 0.05, 0.95)
    return pd.DataFrame({
        "event_date": pd.date_range("2018-01-01", periods=n, freq="3D"),
        "a_skill": a_skill,
        "b_skill": b_skill,
        "diff_skill": a_skill - b_skill,
        "a_implied_prob": a_prob,
        "b_implied_prob": 1.0 - a_prob,
        "diff_implied_prob": 2.0 * a_prob - 1.0,
        "target": target,
    })


_FEATURES = [
    "a_skill", "b_skill", "diff_skill",
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
]
_FAST_PARAMS = {
    "n_estimators": 30, "max_depth": 3, "learning_rate": 0.1,
    "random_state": 42, "eval_metric": "logloss", "use_label_encoder": False,
}


def test_temporal_holdout_refit_serves_full_data_booster():
    frame = _synthetic_training_frame()
    holdout = train_xgboost(
        frame, _FEATURES, calibrate=True, xgb_params=_FAST_PARAMS,
        calibration_method="sigmoid", calibration_cv="temporal_holdout",
    )
    refit = train_xgboost(
        frame, _FEATURES, calibrate=True, xgb_params=_FAST_PARAMS,
        calibration_method="sigmoid", calibration_cv="temporal_holdout_refit",
    )

    assert isinstance(refit["model"], HoldoutCalibratedRefitModel)
    # The served booster is the full-data raw model, not the inner booster.
    assert refit["model"].base_estimator is refit["raw_model"]

    X = frame[_FEATURES].to_numpy()
    p_holdout = holdout["model"].predict_proba(X)[:, 1]
    p_refit = refit["model"].predict_proba(X)[:, 1]
    assert p_refit.shape == p_holdout.shape
    assert np.all((p_refit >= 0) & (p_refit <= 1))
    # Different boosters -> different probabilities somewhere.
    assert np.abs(p_refit - p_holdout).max() > 1e-6

    # The calibrator mapping itself is shared: applying the refit wrapper's
    # calibrator to the holdout booster's raw outputs reproduces the holdout
    # model's calibrated outputs.
    inner_raw = holdout["model"].calibrated_classifiers_[0].estimator.predict_proba(X)[:, 1]
    remapped = refit["model"]._calibrator().predict(inner_raw)
    np.testing.assert_allclose(remapped, p_holdout, atol=1e-9)


def test_temporal_holdout_weighted_changes_calibrator():
    frame = _synthetic_training_frame()
    plain = train_xgboost(
        frame, _FEATURES, calibrate=True, xgb_params=_FAST_PARAMS,
        calibration_method="sigmoid", calibration_cv="temporal_holdout",
        time_decay_half_life_days=365,
    )
    weighted = train_xgboost(
        frame, _FEATURES, calibrate=True, xgb_params=_FAST_PARAMS,
        calibration_method="sigmoid", calibration_cv="temporal_holdout_weighted",
        time_decay_half_life_days=365,
    )
    X = frame[_FEATURES].to_numpy()
    p_plain = plain["model"].predict_proba(X)[:, 1]
    p_weighted = weighted["model"].predict_proba(X)[:, 1]
    # Weights span several e-folds over the cal window — the sigmoid must move.
    assert np.abs(p_plain - p_weighted).max() > 1e-9


def test_antithetic_noise_preserves_no_vig_identities():
    frame = _synthetic_training_frame(n=200)
    frame["_mirror_group_id"] = np.repeat(np.arange(100), 2)
    frame["_mirror_augmented"] = np.tile([0, 1], 100)
    # Make row pairs true mirrors of each other on the odds columns.
    a = frame["a_implied_prob"].to_numpy().copy()
    mirrored_mask = frame["_mirror_augmented"].to_numpy() == 1
    originals = a[~mirrored_mask]
    a[mirrored_mask] = 1.0 - originals
    frame["a_implied_prob"] = a
    frame["b_implied_prob"] = 1.0 - a
    frame["diff_implied_prob"] = 2.0 * a - 1.0

    X = frame[_FEATURES].to_numpy()
    noised = _add_antithetic_odds_noise(X, _FEATURES, frame, noise_std=0.06, seed=11)

    a_idx, b_idx, d_idx = (_FEATURES.index(c) for c in
                           ("a_implied_prob", "b_implied_prob", "diff_implied_prob"))
    # Identities hold exactly on every row.
    np.testing.assert_allclose(noised[:, a_idx] + noised[:, b_idx], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        noised[:, d_idx], noised[:, a_idx] - noised[:, b_idx], atol=1e-12,
    )
    # Mirror consistency: the mirrored row's a equals 1 - original row's a
    # (same shock, opposite sign), away from the clipping boundary.
    orig = noised[~mirrored_mask, a_idx]
    mirr = noised[mirrored_mask, a_idx]
    interior = (orig > 0.03) & (orig < 0.97) & (mirr > 0.03) & (mirr < 0.97)
    np.testing.assert_allclose(mirr[interior], 1.0 - orig[interior], atol=1e-12)
    # Noise was actually applied.
    assert np.abs(noised[:, a_idx] - X[:, a_idx]).max() > 1e-4


def test_antithetic_noise_keeps_nan_rows_nan():
    frame = _synthetic_training_frame(n=50)
    frame.loc[frame.index[:10], ["a_implied_prob", "b_implied_prob", "diff_implied_prob"]] = np.nan
    X = frame[_FEATURES].to_numpy()
    noised = _add_antithetic_odds_noise(X, _FEATURES, frame, noise_std=0.06, seed=7)
    a_idx = _FEATURES.index("a_implied_prob")
    b_idx = _FEATURES.index("b_implied_prob")
    assert np.isnan(noised[:10, a_idx]).all()
    assert np.isnan(noised[:10, b_idx]).all()
