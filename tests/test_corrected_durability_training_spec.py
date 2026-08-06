import hashlib
import json
from dataclasses import asdict

from src.model import training_spec


SPEC_NAME = "full_live_contract_v6_durability_corrected_20260805_fullfit"
SELECTED_EVALUATION_PAYLOAD_SHA256 = (
    "68f2fd6d851224ab395fe469b17a9974d87b8b48d812e2108636d6b889352f45"
)
DOCUMENTED_FULLFIT_FIELDS = {"name", "description", "train_cutoff_date"}


def _effective_training_payload(spec: training_spec.NamedModelTrainingSpec) -> dict:
    payload = asdict(spec)
    if payload["odds_noise_seed"] is None:
        payload["odds_noise_seed"] = int(payload["xgb_params"]["random_state"])
    return payload


def _canonical_payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_corrected_durability_fullfit_is_registered_with_versioned_metadata():
    resolved = training_spec.resolve_named_training_spec(SPEC_NAME)

    assert resolved.name == SPEC_NAME
    assert resolved.train_cutoff_date == "2027-01-01"
    assert resolved.xgb_params["random_state"] == 42
    assert resolved.odds_noise_seed == 42
    assert "corrected-data" in resolved.description.lower()
    assert "small predictive lift" in resolved.description.lower()
    assert "seed-sensitive betting results" in resolved.description.lower()
    assert "p=0.00" not in resolved.description


def test_selected_evaluation_contract_payload_is_pinned_to_comparison_recipe():
    evaluation = training_spec.full_live_contract_v6_durability_spec()

    assert _canonical_payload_sha256(asdict(evaluation)) == (
        SELECTED_EVALUATION_PAYLOAD_SHA256
    )


def test_corrected_durability_fullfit_only_changes_documented_contract_fields():
    evaluation = training_spec.full_live_contract_v6_durability_spec()
    fullfit = training_spec.resolve_named_training_spec(SPEC_NAME)

    evaluation_payload = _effective_training_payload(evaluation)
    fullfit_payload = _effective_training_payload(fullfit)
    changed_fields = {
        field_name
        for field_name in evaluation_payload
        if evaluation_payload[field_name] != fullfit_payload[field_name]
    }

    assert changed_fields == DOCUMENTED_FULLFIT_FIELDS
    assert fullfit.feature_cols == evaluation.feature_cols
    assert fullfit.xgb_params == evaluation.xgb_params
    assert fullfit.calibration_cv == evaluation.calibration_cv == "temporal_holdout"
    assert fullfit.time_decay_half_life == evaluation.time_decay_half_life
    assert fullfit.odds_noise_std == evaluation.odds_noise_std
    assert fullfit.odds_noise_mode == evaluation.odds_noise_mode
