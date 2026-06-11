"""Tests for the SweepTargetSpec and adapters in selection_gate."""

import pytest
from src.strategy.selection_gate import (
    CandidateResult,
    SweepTargetSpec,
    specs_from_variant_names,
)

def test_from_candidate():
    """Verify conversion from CandidateResult to SweepTargetSpec."""
    candidate = CandidateResult(
        name="xgboost_v2",
        dataset_variant="append_only_2026",
        feature_family="production_betsapi",
        calibration="sigmoid",
        metrics={"accuracy": 0.65}
    )
    
    spec = SweepTargetSpec.from_candidate(candidate)
    
    assert spec.model_variant == "xgboost_v2"
    assert spec.dataset_variant == "append_only_2026"
    assert spec.feature_family == "production_betsapi"
    assert spec.calibration_method == "sigmoid"
    # ID format: name_dataset_feature
    assert spec.candidate_id == "xgboost_v2_append_only_2026_production_betsapi"

def test_specs_from_variant_names():
    """Verify conversion of a list of names to SweepTargetSpecs with defaults."""
    names = ["baseline", "combined_best"]
    specs = specs_from_variant_names(names)
    
    assert len(specs) == 2
    assert specs[0].model_variant == "baseline"
    assert specs[1].model_variant == "combined_best"
    
    # Defaults
    assert all(s.dataset_variant == "append_only_2026" for s in specs)
    assert all(s.feature_family == "production" for s in specs)
    # The baseline's default calibration follows the PROMOTED spec (E6).
    from src.model.training_spec import resolve_named_training_spec
    from src.strategy.model_variants import _promoted_spec_name
    promoted = resolve_named_training_spec(_promoted_spec_name())
    assert specs[0].calibration_method == promoted.calibration_method
    assert specs[1].calibration_method == "sigmoid"
    assert all(s.retrain_months == 4 for s in specs)

def test_specs_from_variant_names_custom_defaults():
    """Verify conversion with custom defaults."""
    names = ["variant_x"]
    specs = specs_from_variant_names(
        names,
        dataset_variant="custom_data",
        feature_family="custom_features",
        calibration_method="custom_cal"
    )
    
    s = specs[0]
    assert s.dataset_variant == "custom_data"
    assert s.feature_family == "custom_features"
    assert s.calibration_method == "custom_cal"
    assert s.candidate_id == "variant_x_custom_data_custom_features"

def test_candidate_id_format():
    """Ensure candidate_id matches the expected format."""
    candidate = CandidateResult(
        name="a",
        dataset_variant="b",
        feature_family="c",
        calibration="d"
    )
    spec = SweepTargetSpec.from_candidate(candidate)
    assert spec.candidate_id == "a_b_c"
