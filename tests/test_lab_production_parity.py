"""E6: the model lab must train and infer under the production regime."""

import numpy as np
import pandas as pd
import pytest

from src.model.training_spec import resolve_named_training_spec
from src.strategy import model_variants
from src.strategy.model_lab import _predict_batch_with_model
from src.strategy.model_variants import VariantConfig, train_variant_model


def test_lab_baseline_mirrors_promoted_spec():
    """The lab baseline must resolve to the promoted bundle's spec fields."""
    spec = resolve_named_training_spec(model_variants._promoted_spec_name())
    baseline = model_variants.baseline()

    assert baseline.feature_cols == list(spec.feature_cols)
    assert baseline.calibration_method == spec.calibration_method
    assert baseline.calibration_cv == spec.calibration_cv
    assert baseline.xgb_params == spec.xgb_params
    assert baseline.time_decay_half_life == spec.time_decay_half_life
    assert baseline.odds_noise_std == spec.odds_noise_std
    assert getattr(baseline, "_native_nan", False) == (spec.impute_strategy == "native_nan")


def test_train_variant_model_routes_through_production_trainer(monkeypatch):
    """Non-indicator variants must train via train_xgboost (mirror augmentation)."""
    calls = {}

    def fake_train_xgboost(train_df, feature_cols, **kwargs):
        calls["kwargs"] = kwargs
        calls["n_rows"] = len(train_df)
        return {"model": object(), "feature_cols": feature_cols,
                "impute_strategy": kwargs.get("impute_strategy")}

    import src.model.train as train_module
    monkeypatch.setattr(train_module, "train_xgboost", fake_train_xgboost)

    variant = VariantConfig(
        name="t", description="t",
        calibration_method="sigmoid",
        calibration_cv="temporal_holdout",
        odds_noise_std=0.06,
        time_decay_half_life=365,
    )
    variant._native_nan = True

    train_df = pd.DataFrame({
        "target": [1, 0], "a_x": [1.0, 2.0], "b_x": [2.0, 1.0],
        "event_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
    })
    result = train_variant_model(train_df, ["a_x", "b_x"], variant)

    assert calls["kwargs"]["impute_strategy"] == "native_nan"
    assert calls["kwargs"]["calibration_method"] == "sigmoid"
    assert calls["kwargs"]["calibration_cv"] == "temporal_holdout"
    assert calls["kwargs"]["odds_noise_std"] == 0.06
    assert calls["kwargs"]["time_decay_half_life_days"] == 365
    assert result["impute_strategy"] == "native_nan"


def test_lab_inference_is_orientation_symmetric():
    """Lab native-NaN predictions must average both A/B orientations."""

    class AsymmetricModel:
        """Predicts higher prob_a the larger a_x is — orientation-sensitive."""

        def predict_proba(self, X):
            p = 1.0 / (1.0 + np.exp(-(X[:, 0] - X[:, 1])))
            # Inject deliberate asymmetry so a single orientation differs
            # from the symmetrized average.
            p = np.clip(p + 0.10, 0.0, 1.0)
            return np.column_stack([1.0 - p, p])

    feature_cols = ["a_x", "b_x", "diff_x"]
    frame = pd.DataFrame({
        "a_x": [3.0], "b_x": [1.0], "diff_x": [2.0],
    })
    model_result = {
        "model": AsymmetricModel(),
        "feature_cols": feature_cols,
        "col_medians": np.zeros(3),
        "impute_strategy": "native_nan",
    }
    result = _predict_batch_with_model(frame, model_result)
    prob_a = float(result["prob_a"].iloc[0])

    # Single original orientation: sigmoid(2) + 0.1 ≈ 0.981
    # Swapped orientation: sigmoid(-2) + 0.1 ≈ 0.219 → 1 - 0.219 = 0.781
    # Symmetrized: (0.981 + 0.781) / 2 ≈ 0.881
    single = 1.0 / (1.0 + np.exp(-2.0)) + 0.10
    swapped = 1.0 / (1.0 + np.exp(2.0)) + 0.10
    expected = (single + (1.0 - swapped)) / 2.0
    assert prob_a == pytest.approx(expected, abs=1e-9)
    assert prob_a != pytest.approx(single, abs=1e-3)
    assert float(result["prob_b"].iloc[0]) == pytest.approx(1.0 - prob_a, abs=1e-12)
